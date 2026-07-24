"""Новое ядро обработки архивов с получением подзаказов через API sborka.ua.

Поток обработки:
1. найти RAR/ZIP во входном каталоге и безопасно распаковать;
2. найти номер основного заказа в имени распакованной папки;
3. получить подзаказы запросом ``action=getSubOrders`` и добавить основной заказ;
4. проверить количество макетов и атомарно переименовать их;
5. переместить исходный архив в ``_DONE_`` либо папку заказа в ручную проверку.
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
from local_llm import LlmProposal, LocalLlmError, request_side_proposal
from pdf_pages import PdfPageCountError, count_layout_units
from unpack import unpack_archives
from website_parser import OrderDataError, SiteAccessError, fetch_suborders


DONE_MARKER = ".done"
CONFLICT_MARKER = ".conflict_pending"
DONE_DIR = "_DONE_"
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
    skipped: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


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
    _, back_side = sides.split("-", maxsplit=1)
    return 2 if int(back_side) > 0 else 1


def _name_for_side(base_name: str, side: str) -> str:
    pattern = r"([_-]?)(\d*)-?(face|back)"

    def replace_side(match: re.Match) -> str:
        prefix, number = match.group(1), match.group(2)
        return f"{prefix}{number + '-' if number else ''}{side}"

    if re.search(pattern, base_name, flags=re.IGNORECASE):
        return re.sub(pattern, replace_side, base_name, flags=re.IGNORECASE)
    return f"{base_name}-{side}"


def _layout_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"\d+", path.name)
    if match:
        return 0, int(match.group()), path.name.casefold()
    return 1, 0, path.name.casefold()


def _group_files_by_order(
    files: list[Path],
    page_details: dict[Path, int],
    order_count: int,
    sides_per_order: int,
) -> list[list[Path]]:
    """Группирует файлы по заказам, считая страницы PDF печатными сторонами."""
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
        files, page_details, len(order_ids), per_suborder
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
    for order_index, suborder in enumerate(order_ids):
        batch = groups[order_index]
        is_container = (
            len(batch) == 1 and page_details[batch[0]] == per_suborder
        )
        base_name = folder.name.replace(order_number, suborder)

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
            if is_container:
                new_name = f"{base_name}_{order_index + 1}{source.suffix}"
            elif per_suborder == 2:
                side = "face" if side_index == 0 else "back"
                new_name = _name_for_side(base_name, side) + source.suffix
            else:
                # Сохраняем действующее правило для односторонних подзаказов.
                new_name = f"{base_name}_{order_index + 1}{source.suffix}"
            operations.append((source, source.with_name(new_name)))

    return operations


def _find_source_archive(source_dir: Path, folder_name: str) -> Path | None:
    for candidate in source_dir.iterdir():
        if (
            candidate.is_file()
            and candidate.stem == folder_name
            and candidate.suffix.casefold() in {".rar", ".zip"}
        ):
            return candidate
    return None


def _unique_destination(directory: Path, filename: str) -> Path:
    destination = directory / filename
    if not destination.exists():
        return destination
    source = Path(filename)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return directory / f"{source.stem}_{timestamp}{source.suffix}"


def _mark_done(folder: Path, source_dir: Path) -> None:
    (folder / DONE_MARKER).write_text(
        f"Обработано через API: {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    archive = _find_source_archive(source_dir, folder.name)
    if archive is None:
        return
    done_dir = source_dir / DONE_DIR
    done_dir.mkdir(exist_ok=True)
    shutil.move(str(archive), str(_unique_destination(done_dir, archive.name)))


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
    shutil.move(str(folder), str(destination))
    (destination / "_PROBLEM.txt").write_text(
        f"Дата: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Папка: {folder.name}\n"
        f"Причина: {reason}\n",
        encoding="utf-8",
    )
    archive = _find_source_archive(source_dir, folder.name)
    if archive is not None:
        archive_manual_dir = source_dir / MANUAL_CHECK_DIR
        archive_manual_dir.mkdir(exist_ok=True)
        shutil.move(
            str(archive),
            str(_unique_destination(archive_manual_dir, archive.name)),
        )


def process_archives_with_api(
    source_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    api_key: str | None = None,
    timeout: int = 10,
    use_local_llm: bool = False,
    confirm_llm=None,
) -> ProcessingSummary:
    """Распаковывает и обрабатывает все заказы в указанных каталогах."""
    source_path = Path(source_dir).expanduser().resolve()
    output_path = Path(output_dir or source_dir).expanduser().resolve()
    _prepare_directory(source_path, "Входной каталог")
    if output_path != source_path:
        _prepare_directory(output_path, "Выходной каталог")

    unpack_archives(str(source_path), str(output_path))
    summary = ProcessingSummary()

    for folder in sorted(output_path.iterdir(), key=lambda item: item.name.casefold()):
        if not folder.is_dir() or folder.name.startswith(("_", ".")):
            continue
        if (folder / DONE_MARKER).exists():
            summary.skipped.append(folder.name)
            continue
        if (folder / CONFLICT_MARKER).exists():
            summary.skipped.append(folder.name)
            continue

        print(f"\n[API] Обработка: {folder.name}")
        order_number = parse_filename(folder.name).get("order_number")
        if not order_number:
            reason = "В имени папки не найден номер основного заказа"
            _move_to_manual_check(folder, source_path, output_path, reason)
            summary.manual_check.append(f"{folder.name}: {reason}")
            continue

        try:
            suborders = fetch_suborders(order_number, api_key=api_key, timeout=timeout)
            # getSubOrders возвращает только дочерние номера. Основной заказ
            # соответствует первому комплекту макетов и добавляется отдельно.
            order_ids = [order_number]
            order_ids.extend(item for item in suborders if item != order_number)
            print(f"    🌐 Итоговый список заказов: {', '.join(order_ids)}")
            files = list_layout_files(folder)
            print(f"    📄 Найдено файлов макетов: {len(files)}")
            layout_units, page_details = count_layout_units(files)
            for path, pages in page_details.items():
                if path.suffix.casefold() == ".pdf":
                    print(
                        f"       PDF {path.relative_to(folder)}: "
                        f"{pages} стр."
                    )
            print(f"    📄 Всего печатных страниц/сторон: {layout_units}")
            proposal = None
            collected_assignments = []
            proposal_confidences = []
            proposal_reasons = []
            while True:
                try:
                    operations = build_api_mapping(
                        folder, files, order_ids, llm_proposal=proposal
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
                    print(
                        f"    🤖 LM Studio, заказ {ambiguity.order_id}: "
                        f"{', '.join(relative_files)}"
                    )
                    partial = request_side_proposal(
                        folder.name, relative_files, [ambiguity.order_id]
                    )
                    print(
                        f"    🤖 LM Studio: уверенность {partial.confidence:.0%}; "
                        f"{partial.reason or 'объяснение не указано'}"
                    )
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
                    raise ApiCoreError("оператор не подтвердил предложение LM Studio")
                print("    ✅ Оператор подтвердил предложение")

            print("    🔄 План переименования:")
            for source, destination in operations:
                print(f"       {source.name}  →  {destination.name}")
            atomic_rename_many(operations)
            _write_rename_log(folder, operations, output_path / "rename_log.txt")
            _mark_done(folder, source_path)
            summary.processed.append(folder.name)
            for old_path, new_path in operations:
                print(f"    {old_path.name} -> {new_path.name}")
        except SiteAccessError as exc:
            # Сетевые ошибки должны оставлять заказ на месте для безопасного повтора.
            summary.failures.append(f"{folder.name}: {exc}")
            print(f"    ❌ {exc}")
        except (ApiCoreError, OrderDataError, LocalLlmError, PdfPageCountError) as exc:
            _move_to_manual_check(folder, source_path, output_path, str(exc))
            summary.manual_check.append(f"{folder.name}: {exc}")
            print(f"    ⚠️ Ручная проверка: {exc}")
        except Exception as exc:
            summary.failures.append(f"{folder.name}: {exc}")
            print(f"    ❌ Непредвиденная ошибка: {exc}")

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
    args = parser.parse_args()

    summary = process_archives_with_api(
        args.source_dir,
        args.output_dir,
        timeout=args.timeout,
        use_local_llm=args.use_local_llm,
    )
    print(
        "\nИтог: "
        f"обработано={len(summary.processed)}, "
        f"ручная проверка={len(summary.manual_check)}, "
        f"пропущено={len(summary.skipped)}, "
        f"ошибки={len(summary.failures)}"
    )
    return 0 if summary.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
