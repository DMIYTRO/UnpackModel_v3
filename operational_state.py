"""Надёжные примитивы для возобновляемой обработки заказов.

Модуль не знает деталей бизнес-процесса. Он предоставляет:

* атомарный JSON-журнал состояния одного заказа;
* lock-файлы запуска и заказа с явным управлением устаревшими блокировками;
* уникальные пути, пригодные для временных файлов и каталогов.
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class OperationalStateError(RuntimeError):
    """Базовая ошибка эксплуатационного состояния."""


class InvalidStageTransition(OperationalStateError):
    """Запрошен недопустимый переход между стадиями."""


class LockHeldError(OperationalStateError):
    """Блокировка уже удерживается другим процессом."""

    def __init__(self, path: Path, owner: Mapping[str, Any] | None, *, stale: bool):
        self.path = path
        self.owner = owner
        self.stale = stale
        status = "устарела" if stale else "активна"
        super().__init__(f"Блокировка {path} уже существует ({status})")


class OrderStage(str, Enum):
    PREPARED = "prepared"
    RENAMED = "renamed"
    ARCHIVED = "archived"
    DONE = "done"
    MANUAL_REVIEW = "manual_review"
    FAILED = "failed"


_ALLOWED_TRANSITIONS: dict[OrderStage, frozenset[OrderStage]] = {
    OrderStage.PREPARED: frozenset(
        {OrderStage.RENAMED, OrderStage.MANUAL_REVIEW, OrderStage.FAILED}
    ),
    OrderStage.RENAMED: frozenset(
        {OrderStage.ARCHIVED, OrderStage.MANUAL_REVIEW, OrderStage.FAILED}
    ),
    OrderStage.ARCHIVED: frozenset({OrderStage.DONE, OrderStage.FAILED}),
    OrderStage.DONE: frozenset(),
    OrderStage.MANUAL_REVIEW: frozenset(),
    OrderStage.FAILED: frozenset(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OrderState:
    order_id: str
    stage: OrderStage
    updated_at: str
    run_id: str
    details: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OrderState":
        try:
            return cls(
                order_id=str(value["order_id"]),
                stage=OrderStage(value["stage"]),
                updated_at=str(value["updated_at"]),
                run_id=str(value["run_id"]),
                details=dict(value.get("details", {})),
                version=int(value.get("version", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OperationalStateError("Некорректный файл состояния заказа") from exc


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    """Записать JSON атомарно через временный файл в том же каталоге."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = unique_path(destination.parent, prefix=f"{destination.name}.", suffix=".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        # Зафиксировать изменение записи каталога там, где это поддерживается.
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class OrderStateStore:
    """JSON-журнал состояния одного заказа."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> OrderState | None:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OperationalStateError(
                f"Повреждён файл состояния: {self.path}"
            ) from exc
        if not isinstance(value, dict):
            raise OperationalStateError(f"Некорректный файл состояния: {self.path}")
        return OrderState.from_dict(value)

    def prepare(
        self,
        order_id: str,
        *,
        run_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> OrderState:
        if self.path.exists():
            raise OperationalStateError(f"Состояние уже существует: {self.path}")
        state = OrderState(
            order_id=str(order_id),
            stage=OrderStage.PREPARED,
            updated_at=_utc_now(),
            run_id=run_id or uuid.uuid4().hex,
            details=dict(details or {}),
        )
        self.save(state)
        return state

    def transition(
        self,
        stage: OrderStage | str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> OrderState:
        current = self.load()
        if current is None:
            raise OperationalStateError(f"Состояние не найдено: {self.path}")
        target = OrderStage(stage)
        if target == current.stage:
            return current
        if target not in _ALLOWED_TRANSITIONS[current.stage]:
            raise InvalidStageTransition(
                f"Недопустимый переход {current.stage.value} → {target.value}"
            )
        merged_details = dict(current.details)
        if details:
            merged_details.update(details)
        updated = OrderState(
            order_id=current.order_id,
            stage=target,
            updated_at=_utc_now(),
            run_id=current.run_id,
            details=merged_details,
            version=current.version,
        )
        self.save(updated)
        return updated

    def save(self, state: OrderState) -> None:
        payload = asdict(state)
        payload["stage"] = state.stage.value
        atomic_write_json(self.path, payload)


@dataclass(frozen=True)
class LockOwner:
    token: str
    pid: int
    hostname: str
    created_at: str
    created_timestamp: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LockOwner":
        return cls(
            token=str(value["token"]),
            pid=int(value["pid"]),
            hostname=str(value["hostname"]),
            created_at=str(value["created_at"]),
            created_timestamp=float(value["created_timestamp"]),
        )


class FileLock:
    """Межпроцессная блокировка на основе эксклюзивного создания файла.

    Устаревшая блокировка удаляется только при ``break_stale=True``. Возраст
    определяется по времени из файла, а при повреждённом файле — по mtime.
    """

    def __init__(self, path: Path | str, *, stale_after: float | None = None):
        self.path = Path(path)
        self.stale_after = stale_after
        self.owner: LockOwner | None = None

    def inspect(self) -> tuple[LockOwner | None, bool]:
        try:
            stat = self.path.stat()
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None, False
        try:
            owner = LockOwner.from_dict(json.loads(raw))
            created = owner.created_timestamp
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            owner = None
            created = stat.st_mtime
        stale = (
            self.stale_after is not None
            and max(0.0, time.time() - created) >= self.stale_after
        )
        return owner, stale

    def acquire(self, *, break_stale: bool = False) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner = LockOwner(
            token=uuid.uuid4().hex,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            created_at=_utc_now(),
            created_timestamp=time.time(),
        )
        payload = json.dumps(asdict(owner), ensure_ascii=False, sort_keys=True) + "\n"

        while True:
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                existing, stale = self.inspect()
                if not stale or not break_stale:
                    raise LockHeldError(self.path, asdict(existing) if existing else None, stale=stale)
                # Запоминаем inode: не удаляем файл, если между inspect и unlink
                # другой процесс уже заменил блокировку.
                try:
                    before = self.path.stat()
                    current, current_stale = self.inspect()
                    after = self.path.stat()
                except FileNotFoundError:
                    continue
                if (
                    before.st_ino != after.st_ino
                    or not current_stale
                    or (existing and current and existing.token != current.token)
                ):
                    continue
                self.path.unlink()
                continue
            try:
                os.write(fd, payload.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            self.owner = owner
            return self

    def release(self) -> None:
        if self.owner is None:
            return
        try:
            current, _ = self.inspect()
            if current is None or current.token != self.owner.token:
                raise OperationalStateError(
                    f"Нельзя снять чужую или изменённую блокировку: {self.path}"
                )
            self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.owner = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def run_lock(root: Path | str, **kwargs: Any) -> FileLock:
    """Создать блокировку всего запуска в указанном рабочем каталоге."""
    return FileLock(Path(root) / ".unpack-run.lock", **kwargs)


def order_lock(order_directory: Path | str, **kwargs: Any) -> FileLock:
    """Создать блокировку конкретного заказа."""
    return FileLock(Path(order_directory) / ".order.lock", **kwargs)


def unique_path(
    directory: Path | str,
    *,
    prefix: str = "",
    suffix: str = "",
) -> Path:
    """Вернуть практически уникальный UUID-путь, ещё не создавая его."""
    parent = Path(directory)
    while True:
        candidate = parent / f"{prefix}{uuid.uuid4().hex}{suffix}"
        if not candidate.exists():
            return candidate
