"""Безопасное пакетное переименование файлов.

Все исходники и назначения проверяются до первого изменения. Переименование
выполняется через уникальные временные имена, поэтому поддерживает в том числе
обмен именами. При ошибке уже сделанные изменения откатываются.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable


class RenameTransactionError(RuntimeError):
    """Пакет переименований не может быть безопасно выполнен."""


def _normalise(mapping: Iterable[tuple[Path | str, Path | str]]) -> list[tuple[Path, Path]]:
    operations = [(Path(source), Path(destination)) for source, destination in mapping]
    sources = [source.absolute() for source, _ in operations]
    destinations = [destination.absolute() for _, destination in operations]

    if len(set(sources)) != len(sources):
        raise RenameTransactionError("Один исходный файл указан несколько раз")
    if len(set(destinations)) != len(destinations):
        raise RenameTransactionError("Несколько файлов получают одно имя")

    moving_sources = {
        source for source, destination in zip(sources, destinations) if source != destination
    }
    for source, destination in zip(sources, destinations):
        if not source.is_file():
            raise RenameTransactionError(f"Исходный файл не найден: {source}")
        if not destination.parent.is_dir():
            raise RenameTransactionError(f"Папка назначения не найдена: {destination.parent}")
        if destination.exists() and destination != source and destination not in moving_sources:
            raise RenameTransactionError(f"Файл назначения уже существует: {destination}")

    return [pair for pair in zip(sources, destinations) if pair[0] != pair[1]]


def atomic_rename_many(mapping: Iterable[tuple[Path | str, Path | str]]) -> None:
    """Атомарно, с точки зрения приложения, применяет карту переименований.

    Файловая система не предоставляет общей транзакции для нескольких файлов,
    поэтому при системной ошибке функция выполняет best-effort откат и сообщает
    как исходную ошибку, так и возможные ошибки отката.
    """
    operations = _normalise(mapping)
    if not operations:
        return

    staged: list[tuple[Path, Path, Path]] = []
    completed: list[tuple[Path, Path, Path]] = []
    try:
        for source, destination in operations:
            # QNAP/Samba may translate a leading dot into the persistent DOS
            # Hidden attribute.  Use a non-hidden, unsupported-extension name
            # so the final file remains visible to Windows clients.
            temporary = source.with_name(f"rename-{uuid.uuid4().hex}-{source.name}.tmp")
            while temporary.exists():
                temporary = source.with_name(f"rename-{uuid.uuid4().hex}-{source.name}.tmp")
            os.rename(source, temporary)
            staged.append((source, temporary, destination))

        for source, temporary, destination in staged:
            os.rename(temporary, destination)
            completed.append((source, temporary, destination))
    except Exception as exc:
        rollback_errors = []
        for source, temporary, destination in reversed(completed):
            try:
                if destination.exists():
                    os.rename(destination, temporary)
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        for source, temporary, _ in reversed(staged):
            try:
                if temporary.exists():
                    os.rename(temporary, source)
            except Exception as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        detail = f"; ошибки отката: {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise RenameTransactionError(f"Пакетное переименование отменено: {exc}{detail}") from exc
