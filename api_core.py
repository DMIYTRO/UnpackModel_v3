"""Новое ядро обработки архивов с получением подзаказов через API sborka.ua.

Поток обработки:
1. найти RAR/ZIP во входном каталоге и безопасно распаковать;
2. найти номер основного заказа в имени распакованной папки;
3. получить подзаказы запросом ``action=getSubOrders`` и добавить основной заказ;
4. проверить количество макетов и атомарно переименовать их;
5. сохранить исходный архив в папке заказа и при необходимости перенести весь
   заказ в ручную проверку.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from atomic_rename import atomic_rename_many
from classifier import classify_face_back_paths
from file_discovery import list_layout_files
from filename_parser import parse_filename
from local_llm import (
    LlmProposal,
    LocalLlmError,
    request_side_proposal,
    request_vision_side_proposal,
)
from operational_state import (
    LockHeldError,
    OrderStage,
    OrderStateStore,
    run_lock,
    unique_path,
)
from pdf_pages import PdfPageCountError, count_layout_units
from unpack import SOURCE_ARCHIVE_DIR, unpack_archives
from website_parser import OrderDataError, SiteAccessError, fetch_suborders


DONE_MARKER = ".done"
STATE_FILE = ".processing_state.json"
CONFLICT_MARKER = ".conflict_pending"
MANUAL_CHECK_DIR = "_REQUIRES_MANUAL_CHECK_"


class ApiCoreError(RuntimeError):
    """Ошибка подготовки или обработки заказа новым ядром."""


class AmbiguousSidesError(ApiCoreError):
    """По обычным правилам нельзя безопасно определить лицо и оборот."""

    def __init__(self, message: str, order_id: str, files: list[Path]):
        super().__init__(message)
        self.order_id = order_id
        self.files = files


@dataclass
class ProcessingSummary:
    processed: list[str] = field(default_factory=list)
    manual_check: list[str] = field(default_factory=list)
    manual_reason_counts: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def add_manual(self, folder_name: str, reason: str) -> None:
        self.manual_check.append(f"{folder_name}: {reason}")
        category = _manual_reason_category(reason)
        self.manual_reason_counts[category] = (
            self.manual_reason_counts.get(category, 0) + 1
        )


def _manual_reason_category(reason: str) -> str:
    """Сворачивает технические сообщения в понятные оператору категории."""
    lowered = reason.casefold()
    if "номер" in lowered and "заказ" in lowered:
        return "нет номера заказа"
    if any(
        marker in lowered
        for marker in (
            "лицо и оборот",
            "лицев",
            "оборотн",
            "обратн",
            "face",
            "back",
            "назначения сторон",
            "локальная llm",
            "lm studio",
            "оператор не подтвердил",
        )
    ):
        return "спорные стороны"
    if any(
        marker in lowered
        for marker in (
            "количество",
            "страниц",
            "сторон",
            "комплект",
            "не хватает печатной",
            "сгруппировать макеты",
        )
    ):
        return "несовпадение количества/страниц"
    return "другая причина"


def _network_mount_root(path: Path) -> Path | None:
    if not path.is_absolute():
        return None
    parts = path.parts
    if len(parts) >= 3 and parts[1] in {"mnt", "Volumes"}:
        return Path(parts[0]) / parts[1] / parts[2]
    return None


def _prepare_directory(path: Path, label: str) -> None:
    """Готовит каталог, не маскируя отключённый NAS обычной папкой."""
    mount_root = _network_mount_root(path)
    if mount_root is not None and not os.path.ismount(mount_root):
        raise ApiCoreError(f"{label} недоступен: сетевой ресурс {mount_root} не примонтирован")
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ApiCoreError(f"{label} не является каталогом: {path}")
        next(path.iterdir(), None)
    except ApiCoreError:
        raise
    except OSError as exc:
        raise ApiCoreError(f"{label} недоступен: {path}: {exc}") from exc


def files_per_suborder(folder_name: str) -> int:
    sides = parse_filename(folder_name).get("sides")
    if not sides:
        raise ApiCoreError(f"В имени папки нет сторонности X-Y: {folder_name}")
    front_side, back_side = (int(value) for value in sides.split("-", maxsplit=1))
    if front_side == 0 and back_side == 0:
        raise ApiCoreError(f"Сторонность 0-0 не содержит печатных сторон: {folder_name}")
    return int(front_side > 0) + int(back_side > 0)


def _name_for_side(base_name: str, side: str) -> str:
    # Маркер должен быть самостоятельным сегментом. В частности, не меняем
    # ``interface`` и ``backing``.
    pattern = r"(?<![A-Za-z])([_-]?)(\d*)-?(face|back)(?![A-Za-z])"

    def replace_side(match: re.Match) -> str:
        prefix, number = match.group(1), match.group(2)
        return f"{prefix}{number + '-' if number else ''}{side}"

    if re.search(pattern, base_name, flags=re.IGNORECASE):
        return re.sub(pattern, replace_side, base_name, flags=re.IGNORECASE)
    return f"{base_name}-{side}"


def _replace_order_number(folder_name: str, old: str, new: str) -> str:
    """Заменяет только номер заказа во второй позиции блока ``(N-ORDER)``."""
    pattern = re.compile(rf"(\(\d+-){re.escape(old)}(\))")
    replaced, count = pattern.subn(rf"\g<1>{new}\g<2>", folder_name, count=1)
    if count != 1:
        raise ApiCoreError(
            f"Не удалось однозначно заменить номер заказа {old} в {folder_name}"
        )
    return replaced


def _mentioned_order_ids(path: Path, order_ids: list[str]) -> list[str]:
    """Находит только полные числовые токены заказа в имени файла."""
    text = path.name
    return [
        order_id
        for order_id in order_ids
        if re.search(rf"(?<!\d){re.escape(order_id)}(?!\d)", text)
    ]


def _layout_sort_key(path: Path) -> tuple[int, int, str]:
    """Естественный порядок комплектов: первое число, затем полное имя."""
    match = re.search(r"\d+", path.name)
    if match:
        return 0, int(match.group()), path.as_posix().casefold()
    return 1, 0, path.as_posix().casefold()


def _has_explicit_order_mapping(files: list[Path], order_ids: list[str]) -> bool:
    return bool(files) and all(
        len(_mentioned_order_ids(path, order_ids)) == 1 for path in files
    )


def _group_files_by_order(
    files: list[Path],
    page_details: dict[Path, int],
    order_ids: list[str],
    sides_per_order: int,
    *,
    strict_order_ids: bool = False,
) -> list[list[Path]]:
    """Группирует файлы по заказам, считая страницы PDF печатными сторонами."""
    order_count = len(order_ids)
    if order_count > 1 and _has_explicit_order_mapping(files, order_ids):
        by_order: dict[str, list[Path]] = {order_id: [] for order_id in order_ids}
        for path in sorted(files, key=lambda item: item.as_posix().casefold()):
            mentioned = _mentioned_order_ids(path, order_ids)
            by_order[mentioned[0]].append(path)
        groups = []
        for order_id in order_ids:
            group = by_order[order_id]
            units = sum(page_details[path] for path in group)
            if units != sides_per_order:
                raise ApiCoreError(
                    f"Заказ {order_id}: найдено {units} печатных сторон, "
                    f"ожидалось {sides_per_order}"
                )
            groups.append(group)
        return groups
    if order_count > 1 and strict_order_ids:
        raise ApiCoreError(
            "Для нескольких заказов не во всех именах файлов указан ровно "
            "один полный номер заказа"
        )

    groups = []
    current = []
    current_sides = 0
    for path in sorted(files, key=_layout_sort_key):
        units = page_details[path]
        if units > sides_per_order:
            raise ApiCoreError(
                f"{path.name}: {units} страниц/сторон, но для заказа допустимо "
                f"{sides_per_order}"
            )
        if current_sides + units > sides_per_order:
            raise ApiCoreError(
                f"Не удалось сгруппировать макеты по заказам возле файла {path.name}"
            )
        current.append(path)
        current_sides += units
        if current_sides == sides_per_order:
            groups.append(current)
            current = []
            current_sides = 0
    if current:
        raise ApiCoreError("Последнему заказу не хватает печатной стороны")
    if len(groups) != order_count:
        raise ApiCoreError(
            f"Количество комплектов не совпадает: найдено {len(groups)}, "
            f"ожидалось {order_count}"
        )
    return groups


def build_api_mapping(
    folder: Path,
    files: list[Path],
    order_ids: list[str],
    *,
    llm_proposal: LlmProposal | None = None,
    strict_order_ids: bool = False,
) -> list[tuple[Path, Path]]:
    """Строит полную карту переименования и отклоняет неполные совпадения."""
    order_number = parse_filename(folder.name).get("order_number")
    if not order_number:
        raise ApiCoreError(f"В имени папки нет номера заказа: {folder.name}")
    if not order_ids:
        raise ApiCoreError(f"Не получен список заказов для {order_number}")

    per_suborder = files_per_suborder(folder.name)
    page_details = count_layout_units(files)[1]
    groups = _group_files_by_order(
        files,
        page_details,
        order_ids,
        per_suborder,
        strict_order_ids=strict_order_ids,
    )

    proposed = {}
    if llm_proposal is not None:
        by_name = {str(path.relative_to(folder)): path for path in files}
        for assignment in llm_proposal.assignments:
            proposed[assignment.order_id] = [
                by_name[assignment.face],
                by_name[assignment.back],
            ]
    operations: list[tuple[Path, Path]] = []
    side_scheme = parse_filename(folder.name)["sides"]
    back_only = side_scheme is not None and side_scheme.startswith("0-")
    for order_index, suborder in enumerate(order_ids):
        batch = groups[order_index]
        is_container = (
            len(batch) == 1 and page_details[batch[0]] == per_suborder
        )
        base_name = _replace_order_number(folder.name, order_number, suborder)

        if per_suborder == 2 and not is_container:
            if suborder in proposed:
                batch = proposed[suborder]
            else:
                classified = classify_face_back_paths(batch)
                if classified["face"] is None or classified["back"] is None:
                    names = ", ".join(path.name for path in batch)
                    raise AmbiguousSidesError(
                        f"не удалось определить лицо и оборот для заказа {suborder}: {names}",
                        suborder,
                        batch,
                    )
                batch = [classified["face"], classified["back"]]

        for side_index, source in enumerate(batch):
            if back_only:
                new_name = _name_for_side(base_name, "back") + source.suffix
            elif is_container:
                new_name = f"{base_name}_{order_index + 1}{source.suffix}"
            elif per_suborder == 2:
                side = "face" if side_index == 0 else "back"
                new_name = _name_for_side(base_name, side) + source.suffix
            else:
                new_name = f"{base_name}_{order_index + 1}{source.suffix}"
            operations.append((source, source.with_name(new_name)))

    return operations


def _find_source_archive(source_dir: Path, folder_name: str) -> Path | None:
    for candidate in source_dir.iterdir():
        if (
            candidate.is_file()
            and candidate.stem == folder_name
            and candidate.suffix.casefold() in {".rar", ".zip", ".7z"}
        ):
            return candidate
    return None


def _unique_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    source = Path(filename)
    return unique_path(directory, prefix=f"{source.stem}_", suffix=source.suffix)


def _archive_source(folder: Path, source_dir: Path) -> Path | None:
    """Возвращает архив из папки заказа, мигрируя результат старого запуска."""
    embedded_dir = folder / SOURCE_ARCHIVE_DIR
    embedded = sorted(
        (
            path
            for path in embedded_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in {".rar", ".zip", ".7z"}
        ),
        key=lambda path: path.name.casefold(),
    ) if embedded_dir.is_dir() else []
    if len(embedded) > 1:
        raise ApiCoreError(
            f"В папке {folder.name} найдено несколько исходных архивов"
        )
    if embedded:
        return embedded[0]

    archive = _find_source_archive(source_dir, folder.name)
    if archive is None:
        return None
    embedded_dir.mkdir(exist_ok=True)
    destination = _unique_destination(embedded_dir, archive.name)
    shutil.move(str(archive), str(destination))
    return destination


def _write_done_marker(folder: Path) -> None:
    marker = folder / DONE_MARKER
    temporary = unique_path(folder, prefix=f"{DONE_MARKER}.", suffix=".tmp")
    temporary.write_text(
        f"Обработано через API: {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)


def _finish_renamed_order(
    folder: Path,
    source_dir: Path,
    state_store: OrderStateStore,
) -> None:
    """Завершает архивирование; безопасно вызывается после прерванного запуска."""
    state = state_store.load()
    if state is None:
        raise ApiCoreError(f"Не найдено состояние обработки: {folder.name}")
    if state.stage == OrderStage.RENAMED:
        archived = _archive_source(folder, source_dir)
        state_store.transition(
            OrderStage.ARCHIVED,
            details={"archived_to": str(archived) if archived else None},
        )
        state = state_store.load()
    if state is not None and state.stage == OrderStage.ARCHIVED:
        _write_done_marker(folder)
        state_store.transition(OrderStage.DONE)


def _write_rename_log(
    folder: Path,
    operations: list[tuple[Path, Path]],
    log_path: Path,
) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        for source, destination in operations:
            log_file.write(
                f"[{folder.name}] {source.relative_to(folder)} -> "
                f"{destination.relative_to(folder)}\n"
            )


def _move_to_manual_check(
    folder: Path,
    source_dir: Path,
    output_dir: Path,
    reason: str,
) -> None:
    manual_dir = output_dir / MANUAL_CHECK_DIR
    manual_dir.mkdir(exist_ok=True)
    destination = _unique_destination(manual_dir, folder.name)
    archive = _find_source_archive(source_dir, folder.name)
    if archive is not None:
        _archive_source(folder, source_dir)

    print("    📋 План ручной проверки:")
    print(f"       Категория: {_manual_reason_category(reason)}")
    print(f"       Причина: {reason}")
    print(f"       Папка: {folder}  →  {destination}")
    if archive is not None:
        print(f"       Архив сохранён внутри папки заказа")

    shutil.move(str(folder), str(destination))
    (destination / "_PROBLEM.txt").write_text(
        f"Дата: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Папка: {folder.name}\n"
        f"Причина: {reason}\n",
        encoding="utf-8",
    )


def process_archives_with_api(
    source_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    api_key: str | None = None,
    timeout: int = 10,
    use_local_llm: bool = False,
    confirm_llm=None,
    break_stale_lock: bool = False,
    strict_order_ids: bool = False,
) -> ProcessingSummary:
    """Распаковывает и обрабатывает все заказы в указанных каталогах."""
    source_path = Path(source_dir).expanduser().resolve()
    output_path = Path(output_dir or source_dir).expanduser().resolve()
    _prepare_directory(source_path, "Входной каталог")
    if output_path != source_path:
        _prepare_directory(output_path, "Выходной каталог")

    try:
        process_lock = run_lock(source_path, stale_after=24 * 60 * 60).acquire(
            break_stale=break_stale_lock
        )
    except LockHeldError as exc:
        raise ApiCoreError(str(exc)) from exc

    summary = ProcessingSummary()
    try:
        unpack_archives(str(source_path), str(output_path))

        for folder in sorted(output_path.iterdir(), key=lambda item: item.name.casefold()):
            if not folder.is_dir() or folder.name.startswith(("_", ".")):
                continue
            state_store = OrderStateStore(folder / STATE_FILE)
            state = state_store.load()
            if (folder / DONE_MARKER).exists() or (
                state is not None and state.stage == OrderStage.DONE
            ):
                summary.skipped.append(folder.name)
                continue
            if state is not None and state.stage in {
                OrderStage.RENAMED,
                OrderStage.ARCHIVED,
            }:
                try:
                    _finish_renamed_order(folder, source_path, state_store)
                    summary.processed.append(folder.name)
                except Exception as exc:
                    summary.failures.append(f"{folder.name}: восстановление: {exc}")
                continue
            if (folder / CONFLICT_MARKER).exists():
                summary.skipped.append(folder.name)
                continue

            print(f"\n[API] Обработка: {folder.name}")
            order_number = parse_filename(folder.name).get("order_number")
            if not order_number:
                reason = "В имени папки не найден номер основного заказа"
                _move_to_manual_check(folder, source_path, output_path, reason)
                summary.add_manual(folder.name, reason)
                continue

            try:
                suborders = fetch_suborders(
                    order_number, api_key=api_key, timeout=timeout
                )
                order_ids = [order_number]
                order_ids.extend(item for item in suborders if item != order_number)
                print(f"    🌐 Итоговый список заказов: {', '.join(order_ids)}")
                files = list_layout_files(folder)
                print(f"    📄 Найдено файлов макетов: {len(files)}")
                layout_units, page_details = count_layout_units(files)
                for path, pages in page_details.items():
                    if path.suffix.casefold() == ".pdf":
                        print(f"       PDF {path.relative_to(folder)}: {pages} стр.")
                print(f"    📄 Всего печатных страниц/сторон: {layout_units}")
                if (
                    len(order_ids) > 1
                    and not _has_explicit_order_mapping(files, order_ids)
                    and not strict_order_ids
                ):
                    print(
                        "    🧭 Номера подзаказов есть не во всех именах; "
                        "использую последовательную группировку файлов"
                    )

                proposal = None
                collected_assignments = []
                proposal_confidences = []
                proposal_reasons = []
                while True:
                    try:
                        operations = build_api_mapping(
                            folder,
                            files,
                            order_ids,
                            llm_proposal=proposal,
                            strict_order_ids=strict_order_ids,
                        )
                        break
                    except AmbiguousSidesError as ambiguity:
                        print(f"    ❓ Спорный случай: {ambiguity}")
                        if not use_local_llm:
                            raise ApiCoreError(
                                f"{ambiguity}; локальная LLM не включена"
                            ) from ambiguity
                        relative_files = [
                            str(path.relative_to(folder)) for path in ambiguity.files
                        ]
                        try:
                            print("    🤖 Слой 2: анализ имён файлов")
                            partial = request_side_proposal(
                                folder.name,
                                relative_files,
                                [ambiguity.order_id],
                            )
                        except LocalLlmError as text_error:
                            print(
                                "    👁 Слой 3: имена не дали ответа; "
                                "анализирую изображения"
                            )
                            try:
                                partial = request_vision_side_proposal(
                                    folder.name,
                                    ambiguity.files,
                                    [ambiguity.order_id],
                                    filenames=relative_files,
                                )
                            except LocalLlmError as vision_error:
                                raise LocalLlmError(
                                    f"анализ имён: {text_error}; "
                                    f"анализ изображений: {vision_error}"
                                ) from vision_error
                        collected_assignments.extend(partial.assignments)
                        proposal_confidences.append(partial.confidence)
                        if partial.reason:
                            proposal_reasons.append(partial.reason)
                        proposal = LlmProposal(
                            assignments=tuple(collected_assignments),
                            confidence=min(proposal_confidences),
                            reason="; ".join(proposal_reasons),
                        )

                if proposal is not None:
                    print("    📝 Предлагаемый план переименования:")
                    for source, destination in operations:
                        print(f"       {source.name}  →  {destination.name}")
                    confirmer = confirm_llm or _confirm_llm_proposal
                    if not confirmer(folder, proposal, operations):
                        raise ApiCoreError(
                            "оператор не подтвердил предложение LM Studio"
                        )

                print("    🔄 План переименования:")
                for source, destination in operations:
                    print(f"       {source.name}  →  {destination.name}")
                if state is None:
                    state_store.prepare(
                        order_number,
                        details={
                            "operations": [
                                {
                                    "source": str(source.relative_to(folder)),
                                    "destination": str(
                                        destination.relative_to(folder)
                                    ),
                                }
                                for source, destination in operations
                            ]
                        },
                    )
                atomic_rename_many(operations)
                state_store.transition(OrderStage.RENAMED)
                try:
                    _write_rename_log(
                        folder, operations, output_path / "rename_log.txt"
                    )
                except OSError as exc:
                    print(f"    ⚠️ Не удалось записать журнал переименований: {exc}")
                _finish_renamed_order(folder, source_path, state_store)
                summary.processed.append(folder.name)
                for old_path, new_path in operations:
                    print(f"    {old_path.name} -> {new_path.name}")
            except SiteAccessError as exc:
                # Сетевые ошибки оставляют заказ на месте для безопасного повтора.
                summary.failures.append(f"{folder.name}: {exc}")
                print(f"    ❌ {exc}")
            except (ApiCoreError, OrderDataError, LocalLlmError, PdfPageCountError) as exc:
                _move_to_manual_check(folder, source_path, output_path, str(exc))
                summary.add_manual(folder.name, str(exc))
                print(f"    ⚠️ Ручная проверка: {exc}")
            except Exception as exc:
                summary.failures.append(f"{folder.name}: {exc}")
                print(f"    ❌ Непредвиденная ошибка: {exc}")
    finally:
        process_lock.release()

    return summary


def _confirm_llm_proposal(
    folder: Path,
    proposal: LlmProposal,
    operations: list[tuple[Path, Path]],
) -> bool:
    """Требует явного подтверждения спорного предложения."""
    while True:
        answer = input("    Подтвердить это переименование? [да/нет]: ").strip().casefold()
        if answer in {"да", "д", "yes", "y"}:
            return True
        if answer in {"нет", "н", "no", "n"}:
            return False
        print("    Введите «да» или «нет».")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Распаковать архивы и переименовать макеты через API sborka.ua"
    )
    parser.add_argument("source_dir", nargs="?", default="original_archives")
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument(
        "--use-local-llm",
        action="store_true",
        help="спрашивать LM Studio о спорных двухсторонних заказах",
    )
    parser.add_argument(
        "--break-stale-lock",
        action="store_true",
        help="снять блокировку старше 24 часов и продолжить обработку",
    )
    parser.add_argument(
        "--strict-order-ids",
        action="store_true",
        help=(
            "для нескольких подзаказов запретить последовательную группировку "
            "и требовать номер заказа в каждом имени"
        ),
    )
    args = parser.parse_args()

    summary = process_archives_with_api(
        args.source_dir,
        args.output_dir,
        timeout=args.timeout,
        use_local_llm=args.use_local_llm,
        break_stale_lock=args.break_stale_lock,
        strict_order_ids=args.strict_order_ids,
    )
    print(
        "\nИтог: "
        f"обработано={len(summary.processed)}, "
        f"ручная проверка={len(summary.manual_check)}, "
        f"пропущено={len(summary.skipped)}, "
        f"ошибки={len(summary.failures)}"
    )
    if summary.manual_reason_counts:
        print("Причины ручной проверки:")
        for reason, count in summary.manual_reason_counts.items():
            print(f"  {reason}: {count}")
    return 0 if summary.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
