from __future__ import annotations

import json
import os
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from api_core import (
    ApiCoreError,
    AmbiguousSidesError,
    STATE_FILE,
    _finish_renamed_order,
    _move_to_manual_check,
    build_api_mapping,
)
from atomic_rename import atomic_rename_many
from file_discovery import list_layout_files
from filename_parser import parse_filename
from local_llm import (
    LlmProposal,
    LocalLlmError,
    request_side_proposal,
    request_vision_side_proposal,
)
from operational_state import OrderStage, OrderStateStore, atomic_write_json, order_lock
from unpack import unpack_archives
from website_parser import fetch_suborders


FINAL_ORDER_STATUSES = {"completed", "manual_review", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WebServiceError(RuntimeError):
    pass


class WebPipelineService:
    def __init__(self, state_path: Path, allowed_roots: tuple[Path, ...]):
        self.state_path = state_path
        self.allowed_roots = tuple(path.resolve() for path in allowed_roots)
        self._lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        self.state: dict[str, Any] = {"runs": {}}
        self._load()

    def _load(self) -> None:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("runs"), dict):
                self.state = value
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        for run in self.state["runs"].values():
            if run["status"] in {"queued", "scanning", "analyzing"}:
                run["status"] = "interrupted"
                run["error"] = "Сервис был перезапущен во время анализа"
            for order in run.get("orders", {}).values():
                for item in order.get("items", []):
                    if "proposal_available" in item:
                        continue
                    available = bool(item.get("proposed_name")) and (
                        not order.get("reason")
                        or item.get("proposed_name") != item.get("original_name")
                    )
                    item["proposal_available"] = available
                    if not available:
                        item["proposed_name"] = ""
        self._save()

    def _save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _path(self, value: str) -> Path:
        candidate = Path(value).expanduser().resolve()
        if not any(candidate == root or root in candidate.parents for root in self.allowed_roots):
            raise WebServiceError("Папка находится за пределами разрешённой области")
        if not candidate.is_dir():
            raise WebServiceError(f"Папка не найдена: {candidate}")
        return candidate

    def _event(self, run: dict[str, Any], event_type: str, message: str, **extra: Any) -> None:
        run["events"].append({"id": len(run["events"]) + 1, "type": event_type, "message": message, "created_at": _now(), **extra})
        run["updated_at"] = _now()
        self._save()

    def start(self, options: dict[str, Any]) -> dict[str, Any]:
        source = self._path(str(options.get("source_dir", "")))
        output = self._path(str(options.get("output_dir") or source))
        with self._lock:
            if any(run["status"] in {"queued", "scanning", "analyzing"} for run in self.state["runs"].values()):
                raise WebServiceError("Другой запуск уже выполняется")
            run_id = uuid4().hex
            run = {
                "id": run_id, "status": "queued", "stage": "В очереди", "progress": 0,
                "source_dir": str(source), "output_dir": str(output),
                "use_local_llm": bool(options.get("use_local_llm") or options.get("use_llm")),
                "strict_order_ids": bool(options.get("strict_order_ids") or options.get("strict")),
                "created_at": _now(), "updated_at": _now(), "error": None,
                "events": [], "orders": {},
            }
            self.state["runs"][run_id] = run
            self._cancel[run_id] = threading.Event()
            llm_state = "включена" if run["use_local_llm"] else "выключена"
            self._event(run, "run.queued", f"Запуск поставлен в очередь; LLM {llm_state}")
            threading.Thread(target=self._analyze, args=(run_id,), daemon=True).start()
            return self.public_run(run)

    def _analyze(self, run_id: str) -> None:
        with self._lock:
            run = self.state["runs"][run_id]
            cancel = self._cancel[run_id]
            run.update(status="scanning", stage="Распаковка архивов", progress=5)
            self._event(run, "run.started", "Начата распаковка архивов")
        try:
            unpack_archives(run["source_dir"], run["output_dir"])
            folders = [p for p in sorted(Path(run["output_dir"]).iterdir(), key=lambda p: p.name.casefold()) if p.is_dir() and not p.name.startswith(("_", ".")) and not (p / ".done").exists()]
            total = max(1, len(folders))
            for index, folder in enumerate(folders):
                if cancel.is_set():
                    with self._lock:
                        run.update(status="cancelled", stage="Остановлено")
                        self._event(run, "run.cancelled", "Запуск остановлен оператором")
                    return
                self._analyze_folder(run, folder)
                with self._lock:
                    run["progress"] = 10 + round((index + 1) / total * 85)
                    run["stage"] = f"Проанализировано {index + 1} из {len(folders)}"
                    self._event(run, "scan.progress", run["stage"], progress=run["progress"])
            with self._lock:
                pending = any(order["status"] in {"awaiting_decision", "needs_attention", "postponed"} for order in run["orders"].values())
                run.update(status="awaiting_decisions" if pending else "completed", stage="Требуются решения" if pending else "Готово", progress=100)
                self._event(run, "run.awaiting_decisions" if pending else "run.completed", run["stage"])
        except Exception as exc:
            with self._lock:
                run.update(status="failed", stage="Ошибка", error=str(exc))
                self._event(run, "run.failed", str(exc))

    def _analyze_folder(self, run: dict[str, Any], folder: Path) -> None:
        files = list_layout_files(folder)
        order_number = parse_filename(folder.name).get("order_number")
        operations = []
        reason = ""
        status = "awaiting_decision"
        if not order_number:
            reason, status = "В имени папки не найден номер заказа", "needs_attention"
        else:
            try:
                suborders = fetch_suborders(order_number)
                order_ids = [order_number, *(item for item in suborders if item != order_number)]
                operations = self._mapping_with_optional_llm(
                    folder,
                    files,
                    order_ids,
                    use_local_llm=run["use_local_llm"],
                    strict_order_ids=run["strict_order_ids"],
                )
            except Exception as exc:
                reason, status = str(exc), "needs_attention"
        proposed = {str(source.relative_to(folder)): destination.name for source, destination in operations}
        items = []
        for path in files:
            relative = str(path.relative_to(folder))
            stat = path.stat()
            file_id = uuid4().hex
            items.append({
                "file_id": file_id, "relative_path": relative, "original_name": path.name,
                "proposed_name": proposed.get(relative, ""),
                "proposal_available": relative in proposed,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
        key = uuid4().hex
        run["orders"][key] = {"order_id": key, "folder_name": folder.name, "folder_path": str(folder), "status": status, "reason": reason, "items": items, "created_at": _now()}
        self._event(run, "order.waiting", f"{folder.name}: ожидает решения", order_id=key)

    @staticmethod
    def _mapping_with_optional_llm(
        folder: Path,
        files: list[Path],
        order_ids: list[str],
        *,
        use_local_llm: bool,
        strict_order_ids: bool,
    ) -> list[tuple[Path, Path]]:
        proposal = None
        assignments = []
        confidences = []
        reasons = []
        while True:
            try:
                return build_api_mapping(
                    folder,
                    files,
                    order_ids,
                    llm_proposal=proposal,
                    strict_order_ids=strict_order_ids,
                )
            except AmbiguousSidesError as ambiguity:
                if not use_local_llm:
                    raise ApiCoreError(
                        f"{ambiguity}; локальная LLM не включена"
                    ) from ambiguity
                relative = [str(path.relative_to(folder)) for path in ambiguity.files]
                try:
                    partial = request_side_proposal(
                        folder.name, relative, [ambiguity.order_id]
                    )
                except LocalLlmError:
                    partial = request_vision_side_proposal(
                        folder.name,
                        ambiguity.files,
                        [ambiguity.order_id],
                        filenames=relative,
                    )
                assignments.extend(partial.assignments)
                confidences.append(partial.confidence)
                if partial.reason:
                    reasons.append(partial.reason)
                proposal = LlmProposal(
                    assignments=tuple(assignments),
                    confidence=min(confidences),
                    reason="; ".join(reasons),
                )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.public_run(run) for run in reversed(list(self.state["runs"].values()))]

    def get(self, run_id: str) -> dict[str, Any]:
        try:
            return self.state["runs"][run_id]
        except KeyError as exc:
            raise WebServiceError("Запуск не найден") from exc

    def public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in run.items() if key not in {"source_dir", "output_dir", "orders", "events"}} | {
            "source_dir": run["source_dir"], "output_dir": run["output_dir"],
            "total_orders": len(run["orders"]),
            "pending_orders": sum(item["status"] in {"awaiting_decision", "needs_attention", "postponed"} for item in run["orders"].values()),
        }

    def public_order(self, run_id: str, order: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in order.items() if key != "folder_path"} | {"items": [{key: value for key, value in item.items() if key not in {"relative_path", "size", "mtime_ns"}} | {"preview_url": f"/api/runs/{run_id}/files/{item['file_id']}/preview"} for item in order["items"]]}

    def orders(self, run_id: str) -> list[dict[str, Any]]:
        run = self.get(run_id)
        return [self.public_order(run_id, item) for item in run["orders"].values()]

    def find_order(self, run_id: str, order_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        run = self.get(run_id)
        try:
            return run, run["orders"][order_id]
        except KeyError as exc:
            raise WebServiceError("Заказ не найден") from exc

    def find_file(self, run_id: str, file_id: str) -> Path:
        run = self.get(run_id)
        for order in run["orders"].values():
            for item in order["items"]:
                if item["file_id"] == file_id:
                    root = Path(order["folder_path"]).resolve()
                    path = (root / item["relative_path"]).resolve()
                    if root not in path.parents or not path.is_file():
                        raise WebServiceError("Файл больше недоступен")
                    return path
        raise WebServiceError("Файл не найден")

    @staticmethod
    def _valid_name(name: str, original: str) -> str:
        clean = unicodedata.normalize("NFC", name.strip())
        if not clean or Path(clean).name != clean or clean.startswith(".") or any(ord(c) < 32 for c in clean):
            raise WebServiceError("Новое имя должно быть обычным именем файла без пути")
        if Path(clean).suffix.casefold() != Path(original).suffix.casefold():
            raise WebServiceError("Расширение файла изменять нельзя")
        return clean

    def approve(self, run_id: str, order_id: str, names: dict[str, str]) -> dict[str, Any]:
        with self._lock:
            run, order = self.find_order(run_id, order_id)
            if order["status"] not in {"awaiting_decision", "needs_attention", "postponed"}:
                raise WebServiceError("Решение уже применено")
            folder = Path(order["folder_path"])
            operations = []
            destinations: set[str] = set()
            for item in order["items"]:
                source = folder / item["relative_path"]
                stat = source.stat()
                if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                    raise WebServiceError(f"Файл изменился после анализа: {item['original_name']}")
                new_name = self._valid_name(names.get(item["file_id"], item["proposed_name"]), item["original_name"])
                folded = unicodedata.normalize("NFC", new_name).casefold()
                if folded in destinations:
                    raise WebServiceError(f"Повторяющееся имя: {new_name}")
                destinations.add(folded)
                operations.append((source, source.with_name(new_name)))
            order["status"] = "applying"
            self._event(run, "order.applying", f"Применяется: {order['folder_name']}", order_id=order_id)
        with order_lock(folder):
            store = OrderStateStore(folder / STATE_FILE)
            if store.load() is None:
                store.prepare(order_id, details={"operations": [{"source": str(a.relative_to(folder)), "destination": str(b.relative_to(folder))} for a, b in operations]})
            atomic_rename_many(operations)
            store.transition(OrderStage.RENAMED)
            _finish_renamed_order(folder, Path(run["source_dir"]), store)
        with self._lock:
            order["status"] = "completed"
            self._event(run, "order.completed", f"Готово: {order['folder_name']}", order_id=order_id)
            self._finish_if_ready(run)
            return self.public_order(run_id, order)

    def postpone(self, run_id: str, order_id: str) -> dict[str, Any]:
        with self._lock:
            run, order = self.find_order(run_id, order_id)
            order["status"] = "postponed"
            self._event(run, "order.postponed", f"Отложено: {order['folder_name']}", order_id=order_id)
            return self.public_order(run_id, order)

    def manual_review(self, run_id: str, order_id: str) -> dict[str, Any]:
        with self._lock:
            run, order = self.find_order(run_id, order_id)
            folder = Path(order["folder_path"])
            reason = order["reason"] or "Направлено оператором на ручную проверку"
        _move_to_manual_check(folder, Path(run["source_dir"]), Path(run["output_dir"]), reason)
        with self._lock:
            order["status"] = "manual_review"
            self._event(run, "order.manual_review", f"Ручная проверка: {order['folder_name']}", order_id=order_id)
            self._finish_if_ready(run)
            return self.public_order(run_id, order)

    def _finish_if_ready(self, run: dict[str, Any]) -> None:
        if run["orders"] and all(order["status"] in FINAL_ORDER_STATUSES for order in run["orders"].values()):
            run.update(status="completed", stage="Готово", progress=100)
            self._event(run, "run.completed", "Все заказы обработаны")

    def cancel(self, run_id: str) -> dict[str, Any]:
        run = self.get(run_id)
        self._cancel.setdefault(run_id, threading.Event()).set()
        if run["status"] not in {"scanning", "analyzing", "queued"}:
            run.update(status="cancelled", stage="Остановлено")
            self._event(run, "run.cancelled", "Запуск остановлен оператором")
        return self.public_run(run)


def default_service() -> WebPipelineService:
    project = Path(__file__).resolve().parent.parent
    configured = os.getenv("UNPACK_ALLOWED_ROOTS", str(project))
    roots = tuple(Path(item).expanduser() for item in configured.split(os.pathsep) if item)
    state_path = Path(os.getenv("UNPACK_WEB_STATE_PATH", project / ".web_state" / "state.json"))
    return WebPipelineService(state_path, roots)
