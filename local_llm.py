"""Клиент локальной LLM для спорного распределения лица и оборота."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv


class LocalLlmError(RuntimeError):
    """Локальная модель недоступна или вернула небезопасный ответ."""


@dataclass(frozen=True)
class SideAssignment:
    order_id: str
    face: str
    back: str


@dataclass(frozen=True)
class LlmProposal:
    assignments: tuple[SideAssignment, ...]
    confidence: float
    reason: str


def _extract_json_content(payload: dict) -> dict:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LocalLlmError("LM Studio вернула ответ без текста") from exc
    if not isinstance(content, str) or not content.strip():
        raise LocalLlmError("LM Studio вернула пустой ответ")
    cleaned = content.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LocalLlmError(f"LM Studio вернула невалидный JSON: {content[:200]!r}") from exc
    if not isinstance(result, dict):
        raise LocalLlmError("Ответ LM Studio должен быть JSON-объектом")
    return result


def _validate_proposal(result: dict, filenames: list[str], order_ids: list[str]) -> LlmProposal:
    if result.get("status") == "ambiguous":
        reason = str(result.get("reason") or "Модель не смогла определить стороны")
        raise LocalLlmError(reason)
    assignments = result.get("assignments")
    if not isinstance(assignments, list):
        raise LocalLlmError("В ответе LM Studio нет списка assignments")

    expected_files = set(filenames)
    expected_orders = set(order_ids)
    seen_files: list[str] = []
    seen_orders: list[str] = []
    validated = []
    for item in assignments:
        if not isinstance(item, dict):
            raise LocalLlmError("Элемент assignments должен быть объектом")
        order_id = str(item.get("order_id", ""))
        face = item.get("face")
        back = item.get("back")
        if not isinstance(face, str) or not isinstance(back, str) or face == back:
            raise LocalLlmError(f"Некорректная пара сторон для заказа {order_id!r}")
        seen_orders.append(order_id)
        seen_files.extend((face, back))
        validated.append(SideAssignment(order_id=order_id, face=face, back=back))

    if set(seen_orders) != expected_orders or len(seen_orders) != len(expected_orders):
        raise LocalLlmError("LM Studio вернула неверный набор номеров заказов")
    if set(seen_files) != expected_files or len(seen_files) != len(expected_files):
        raise LocalLlmError("LM Studio пропустила, повторила или придумала имя файла")

    try:
        confidence = float(result.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise LocalLlmError("LM Studio не вернула числовую уверенность") from exc
    if not 0 <= confidence <= 1:
        raise LocalLlmError("Уверенность LM Studio должна быть от 0 до 1")

    return LlmProposal(
        assignments=tuple(validated),
        confidence=confidence,
        reason=str(result.get("reason") or ""),
    )


def request_side_proposal(
    archive_name: str,
    files: list[Path] | list[str],
    order_ids: list[str],
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> LlmProposal:
    """Запрашивает предложение и строго проверяет все имена и номера."""
    load_dotenv()
    server = (base_url or os.getenv("LOCAL_LLM_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
    model_id = model or os.getenv("LOCAL_LLM_MODEL") or "google/gemma-4-e4b"
    request_timeout = timeout or int(os.getenv("LOCAL_LLM_TIMEOUT", "90"))
    filenames = [str(item) if isinstance(item, str) else item.name for item in files]

    instruction = {
        "archive_name": archive_name,
        "order_ids_in_printing_order": order_ids,
        "filenames": filenames,
        "task": (
            "Assign every filename exactly once to face or back of an order. "
            "Use only supplied filenames and order IDs. If evidence is insufficient, "
            "return status=ambiguous. Never guess merely from alphabetical order. "
            "File extension and file type (PDF, TIFF, JPEG, etc.) do not indicate "
            "face or back and must never be used as side evidence."
        ),
        "response_schema": {
            "status": "proposal or ambiguous",
            "assignments": [{"order_id": "string", "face": "filename", "back": "filename"}],
            "confidence": "number from 0 to 1",
            "reason": "short explanation in Russian",
        },
    }
    try:
        response = requests.post(
            f"{server}/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты проверяешь имена печатных макетов. Возвращай только компактный "
                            "валидный JSON без Markdown. При недостатке данных откажись угадывать."
                        ),
                    },
                    {"role": "user", "content": json.dumps(instruction, ensure_ascii=False)},
                ],
                "temperature": 0,
                "max_tokens": 1200,
                "stream": False,
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise LocalLlmError(f"Ошибка обращения к LM Studio: {exc}") from exc

    return _validate_proposal(_extract_json_content(payload), filenames, order_ids)
