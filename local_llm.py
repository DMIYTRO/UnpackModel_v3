"""Клиент локальной LLM для спорного распределения лица и оборота."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - объясняется оператору при вызове
    Image = None
    UnidentifiedImageError = OSError

try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - объясняется оператору при вызове
    pdfium = None


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


VISION_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp", ".pdf"})
VISION_MAX_DIMENSION = 1600
VISION_MAX_PIXELS = 50_000_000


def _response_format(filenames: list[str], order_ids: list[str]) -> dict:
    """Строгая схема ответа для поддерживаемого Ollama JSON mode."""
    generic_assignment = {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "enum": order_ids},
            "face": {"type": "string", "enum": filenames},
            "back": {"type": "string", "enum": filenames},
        },
        "required": ["order_id", "face", "back"],
        "additionalProperties": False,
    }
    if len(order_ids) == 1 and len(filenames) == 2:
        assignment_schema = {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "order_id": {"const": order_ids[0]},
                        "face": {"const": face},
                        "back": {"const": back},
                    },
                    "required": ["order_id", "face", "back"],
                    "additionalProperties": False,
                }
                for face, back in (filenames, list(reversed(filenames)))
            ]
        }
    else:
        assignment_schema = generic_assignment
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "side_proposal",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["proposal", "ambiguous"]},
                    "assignments": {
                        "type": "array",
                        "minItems": len(order_ids),
                        "maxItems": len(order_ids),
                        "items": assignment_schema,
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["status", "assignments", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    }


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
                "response_format": _response_format(filenames, order_ids),
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise LocalLlmError(f"Ошибка обращения к LM Studio: {exc}") from exc

    return _validate_proposal(_extract_json_content(payload), filenames, order_ids)


def _visual_image(path: Path):
    """Открывает изображение или растеризует первую страницу PDF."""
    if path.suffix.casefold() != ".pdf":
        with Image.open(path) as source:
            source.load()
            return source.convert("RGB")
    if pdfium is None:
        raise LocalLlmError(
            "Для визуального анализа PDF не установлена pypdfium2: "
            "python -m pip install -r requirements.txt"
        )
    document = pdfium.PdfDocument(path)
    try:
        if len(document) < 1:
            raise LocalLlmError(f"PDF не содержит страниц: {path.name}")
        page = document[0]
        try:
            width, height = page.get_size()
            longest = max(float(width), float(height), 1.0)
            scale = min(2.5, VISION_MAX_DIMENSION / longest)
            bitmap = page.render(scale=scale)
            try:
                return bitmap.to_pil().convert("RGB").copy()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()


def _image_data_url(path: Path) -> str:
    """Готовит ограниченное JPEG-превью для визуальной модели."""
    if Image is None:
        raise LocalLlmError(
            "Для визуального анализа не установлена Pillow: "
            "python -m pip install -r requirements.txt"
        )
    if path.suffix.casefold() not in VISION_EXTENSIONS:
        raise LocalLlmError(
            f"Визуальный анализ пока не поддерживает формат {path.suffix}: {path.name}"
        )
    try:
        Image.MAX_IMAGE_PIXELS = VISION_MAX_PIXELS
        with _visual_image(path) as image:
            image.thumbnail((VISION_MAX_DIMENSION, VISION_MAX_DIMENSION))
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=88, optimize=True)
    except LocalLlmError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, RuntimeError) as exc:
        raise LocalLlmError(
            f"Не удалось подготовить изображение {path.name}: {exc}"
        ) from exc
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def request_vision_side_proposal(
    archive_name: str,
    files: list[Path],
    order_ids: list[str],
    *,
    filenames: list[str] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> LlmProposal:
    """Третий слой: определяет стороны по уменьшенным изображениям макетов."""
    if len(files) != len(filenames or files):
        raise LocalLlmError("Число файлов и отображаемых имён не совпадает")
    shown_names = filenames or [path.name for path in files]
    if len(files) != 2 or len(order_ids) != 1:
        raise LocalLlmError(
            "Визуальный слой принимает одну пару файлов одного заказа"
        )

    load_dotenv()
    server = (
        base_url
        or os.getenv("LOCAL_LLM_BASE_URL")
        or "http://127.0.0.1:1234/v1"
    ).rstrip("/")
    model_id = (
        model
        or os.getenv("LOCAL_LLM_VISION_MODEL")
        or os.getenv("LOCAL_LLM_MODEL")
        or "google/gemma-4-e4b"
    )
    request_timeout = timeout or int(os.getenv("LOCAL_LLM_TIMEOUT", "90"))

    instruction = {
        "archive_name": archive_name,
        "order_id": order_ids[0],
        "task": (
            "Определи лицо и оборот печатного макета по содержимому изображений. "
            "Логотип/главная идентификация обычно являются лицом; адрес, телефон, "
            "QR-код и подробные контакты обычно являются оборотом. Используй только "
            "указанные имена. Если содержимого недостаточно, верни status=ambiguous."
        ),
        "response_schema": {
            "status": "proposal or ambiguous",
            "assignments": [
                {
                    "order_id": order_ids[0],
                    "face": "точное имя файла",
                    "back": "точное имя файла",
                }
            ],
            "confidence": "number from 0 to 1",
            "reason": "краткое объяснение на русском",
        },
    }
    content: list[dict] = [
        {
            "type": "text",
            "text": json.dumps(instruction, ensure_ascii=False),
        }
    ]
    for name, path in zip(shown_names, files):
        content.append({"type": "text", "text": f"Файл: {name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(path)},
            }
        )

    try:
        response = requests.post(
            f"{server}/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты проверяешь стороны печатных макетов по изображениям. "
                            "Верни только валидный JSON без Markdown."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                "temperature": 0,
                "max_tokens": 2000,
                "stream": False,
                "response_format": _response_format(shown_names, order_ids),
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise LocalLlmError(
            f"Ошибка визуального обращения к LM Studio: {exc}"
        ) from exc

    return _validate_proposal(
        _extract_json_content(payload), shown_names, order_ids
    )
