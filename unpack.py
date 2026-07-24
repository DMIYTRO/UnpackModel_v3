import os
import json
import posixpath
import re
import shutil
import stat
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from datetime import datetime

# Папка для битых архивов
TROUBLES_DIR = "_TROUBLES_"
EXTRACT_TIMEOUT_SECONDS = 15 * 60
MAX_ARCHIVE_SIZE_BYTES = 1_500_000_000
MAX_EXTRACTED_SIZE_BYTES = 1_500_000_000
MAX_ARCHIVE_MEMBERS = 10_000
MAX_COMPRESSION_RATIO = 200
ZIP_UTF8_FLAG = 0x800
CONFLICT_MARKER = ".conflict_pending"
STALE_TEMP_AGE_SECONDS = 24 * 60 * 60


class ArchiveLimitError(RuntimeError):
    """Архив превышает разрешённый размер."""


class ArchiveSafetyError(RuntimeError):
    """Содержимое архива нельзя безопасно распаковать."""


def _unique_path(directory: Path, filename: str) -> Path:
    """Возвращает свободный путь, не перезаписывая существующие данные."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    return directory / f"{Path(filename).stem}_{datetime.now().strftime('%H%M%S')}{Path(filename).suffix}"


def cleanup_stale_extracting_dirs(output_path: Path, *, now: float | None = None) -> list[Path]:
    """Удаляет только старые служебные каталоги незавершённой распаковки."""
    removed = []
    current_time = time.time() if now is None else now
    for candidate in output_path.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(".extracting_"):
            continue
        try:
            if current_time - candidate.stat().st_mtime < STALE_TEMP_AGE_SECONDS:
                continue
            shutil.rmtree(candidate)
            removed.append(candidate)
            print(f"🧹 Удалена зависшая временная папка: {candidate.name}")
        except OSError as exc:
            print(f"⚠️ Не удалось удалить временную папку {candidate.name}: {exc}")
    return removed


def _find_extractor() -> str | None:
    """
    Определяет доступный инструмент для распаковки RAR.
    Возвращает имя утилиты или None если ничего не найдено.

    Поддерживаемые инструменты (в порядке приоритета):
      - unar   (macOS: brew install unar  /  Linux: apt install unar)
      - unrar  (Linux: apt install unrar)
      - 7z     (Linux: apt install p7zip-full  /  macOS: brew install p7zip)
      - 7za, 7zz (альтернативные имена 7-zip)
    """
    for tool in ("unar", "unrar", "7z", "7za", "7zz"):
        if shutil.which(tool):
            return tool
    return None


def _build_command(tool: str, rar_file: Path, extract_dir: Path) -> list[str]:
    """Формирует команду распаковки для конкретного инструмента."""
    if tool == "unar":
        # -f перезаписывать, -D не создавать вложенную папку
        return ["unar", "-f", "-D", "-o", str(extract_dir), str(rar_file)]

    elif tool == "unrar":
        # x — распаковать с путями; -o+ перезаписать; -y без вопросов
        return ["unrar", "x", "-o+", "-y", str(rar_file), str(extract_dir) + os.sep]

    elif tool in ("7z", "7za", "7zz"):
        # x — распаковать; -aoa перезаписать всё; -y без вопросов
        return [tool, "x", str(rar_file), f"-o{extract_dir}", "-aoa", "-y"]

    raise ValueError(f"Неизвестный инструмент: {tool}")


def _validate_archive_member_names(names: list[str], archive_type: str) -> None:
    """Отклоняет абсолютные, выходящие наружу и повторяющиеся пути."""
    destinations: set[str] = set()
    for original_name in names:
        member_name = original_name.replace("\\", "/")
        normalized = posixpath.normpath(member_name)
        if (
            not member_name
            or "\x00" in member_name
            or member_name.startswith("/")
            or re.match(r"^[A-Za-z]:", member_name)
            or normalized in {"", ".", ".."}
            or normalized.startswith("../")
        ):
            raise ArchiveSafetyError(f"{archive_type} содержит небезопасный путь: {original_name}")
        destination_key = normalized.casefold()
        if destination_key in destinations:
            raise ArchiveSafetyError(f"{archive_type} содержит повторяющийся путь: {original_name}")
        destinations.add(destination_key)


def _check_archive_limits(
    members: list[tuple[str, int, int]],
    archive_type: str,
) -> None:
    """Проверяет количество, распакованный объём и коэффициент сжатия."""
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ArchiveLimitError(
            f"{archive_type} содержит слишком много элементов: {len(members)} (лимит {MAX_ARCHIVE_MEMBERS})"
        )
    total_uncompressed = sum(max(size, 0) for _, size, _ in members)
    total_compressed = sum(max(size, 0) for _, _, size in members)
    if total_uncompressed > MAX_EXTRACTED_SIZE_BYTES:
        raise ArchiveLimitError(
            f"Распакованный размер {archive_type} превышает лимит 1,5 ГБ"
        )
    if total_uncompressed and total_compressed == 0:
        raise ArchiveLimitError(f"{archive_type} имеет недостоверный сжатый размер")
    if total_compressed and total_uncompressed / total_compressed > MAX_COMPRESSION_RATIO:
        raise ArchiveLimitError(
            f"Коэффициент распаковки {archive_type} превышает лимит {MAX_COMPRESSION_RATIO}:1"
        )


def _preflight_rar(tool: str, rar_file: Path) -> None:
    """Получает список RAR до извлечения и проверяет его пути и размеры."""
    members: list[tuple[str, int, int]] = []
    if tool == "unar":
        lsar = shutil.which("lsar")
        if not lsar:
            raise ArchiveSafetyError("Для безопасной проверки unar требуется утилита lsar")
        result = subprocess.run(
            [lsar, "-j", str(rar_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=EXTRACT_TIMEOUT_SECONDS,
        )
        payload = json.loads(result.stdout.decode("utf-8"))
        for item in payload.get("lsarContents", []):
            name = item.get("XADFileName")
            if not name:
                continue
            if item.get("XADIsLink") or item.get("XADFileType") in {"SymbolicLink", "HardLink"}:
                raise ArchiveSafetyError(f"RAR содержит ссылку: {name}")
            members.append((name, int(item.get("XADFileSize", 0)), int(item.get("XADCompressedSize", 0))))
    elif tool == "unrar":
        result = subprocess.run(
            ["unrar", "lt", "-c-", "-p-", str(rar_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=EXTRACT_TIMEOUT_SECONDS,
        )
        current: dict[str, str] = {}
        for line in result.stdout.decode("utf-8", errors="strict").splitlines():
            stripped = line.strip()
            if not stripped and current.get("Name"):
                entry_type = current.get("Type", "").casefold()
                if "link" in entry_type:
                    raise ArchiveSafetyError(f"RAR содержит ссылку: {current['Name']}")
                members.append((
                    current["Name"],
                    int(current.get("Size", "0").replace(" ", "")),
                    int(current.get("Packed size", "0").replace(" ", "")),
                ))
                current = {}
                continue
            if ": " in stripped:
                key, value = stripped.split(": ", 1)
                if key in {"Name", "Type", "Size", "Packed size"}:
                    current[key] = value
        if current.get("Name"):
            members.append((
                current["Name"],
                int(current.get("Size", "0").replace(" ", "")),
                int(current.get("Packed size", "0").replace(" ", "")),
            ))
    else:
        result = subprocess.run(
            [tool, "l", "-slt", "-p-", "--", str(rar_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=EXTRACT_TIMEOUT_SECONDS,
        )
        current: dict[str, str] = {}
        in_members = False
        for line in result.stdout.decode("utf-8", errors="strict").splitlines():
            if line.startswith("----------"):
                in_members = True
                continue
            if not in_members:
                continue
            if not line.strip() and current:
                name = current.get("Path")
                if name:
                    if current.get("Symbolic Link") or current.get("Hard Link"):
                        raise ArchiveSafetyError(f"RAR содержит ссылку: {name}")
                    members.append((name, int(current.get("Size") or 0), int(current.get("Packed Size") or 0)))
                current = {}
                continue
            if " = " in line:
                key, value = line.split(" = ", 1)
                current[key] = value
        if current.get("Path"):
            members.append((current["Path"], int(current.get("Size") or 0), int(current.get("Packed Size") or 0)))

    if not members:
        raise ArchiveSafetyError("Не удалось получить список содержимого RAR")
    _validate_archive_member_names([name for name, _, _ in members], "RAR")
    _check_archive_limits(members, "RAR")


def _reject_extracted_links(root: Path) -> None:
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ArchiveSafetyError(f"Архив содержит ссылку: {candidate.relative_to(root)}")


def _legacy_name_score(value: str) -> int:
    """Оценивает, насколько строка похожа на нормальное имя файла."""
    score = 0
    lowercase_seen = any(char.islower() for char in value)
    for index, char in enumerate(value):
        codepoint = ord(char)
        if char.isascii():
            score += 1 if char.isprintable() else -20
        elif 0x0400 <= codepoint <= 0x04FF:
            score += 5
            if lowercase_seen and char.isupper() and index > 0:
                score -= 8
        elif char.isalpha():
            score += 1
        elif char.isspace():
            score += 1
        else:
            score -= 4

    common_pairs = (
        "ст", "но", "на", "ен", "ов", "ни", "пр", "ро", "по", "ли",
        "ре", "та", "ал", "ер", "ти", "те", "ка", "ит", "ан", "ар",
        "ос", "от", "го", "ла", "не", "за", "ва", "де", "ри", "ру",
        "ли", "це", "зв", "во", "ор", "дру", "ма", "ке",
    )
    lowered = value.casefold()
    score += sum(lowered.count(pair) * 3 for pair in common_pairs)
    return score


def decode_legacy_zip_name(filename: str, flag_bits: int) -> str:
    """Исправляет CP866/CP1251-имена ZIP без установленного UTF-8-флага."""
    if flag_bits & ZIP_UTF8_FLAG:
        return filename
    try:
        raw_name = filename.encode("cp437")
    except UnicodeEncodeError:
        return filename

    candidates = [filename]
    # Некоторые архиваторы записывают настоящие UTF-8-байты, но забывают
    # установить ZIP-флаг UTF-8. После стандартного CP437-декодирования это
    # выглядит как ``╤ü╤é...``. Проверяем UTF-8 вместе со старыми Windows/OEM.
    for encoding in ("utf-8", "cp866", "cp1251"):
        try:
            candidate = raw_name.decode(encoding)
        except UnicodeDecodeError:
            continue
        if candidate not in candidates:
            candidates.append(candidate)

    return max(candidates, key=_legacy_name_score)


def _extract_zip_safely(zip_file: Path, extract_dir: Path):
    """Безопасно распаковывает ZIP и восстанавливает старые имена кириллицей."""
    root = extract_dir.resolve()
    destinations: set[Path] = set()
    written_bytes = 0
    with zipfile.ZipFile(zip_file) as archive:
        archive_members = archive.infolist()
        _check_archive_limits(
            [(member.filename, member.file_size, member.compress_size) for member in archive_members],
            "ZIP",
        )
        for member in archive_members:
            member_name = decode_legacy_zip_name(member.filename, member.flag_bits)
            member_name = member_name.replace("\\", "/")
            destination = (root / member_name).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"ZIP содержит небезопасный путь: {member_name}")
            if destination in destinations:
                raise ValueError(f"ZIP содержит повторяющийся путь: {member_name}")
            destinations.add(destination)

            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"ZIP содержит символическую ссылку: {member_name}")
            if member.is_dir() or member_name.endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    written_bytes += len(chunk)
                    if written_bytes > MAX_EXTRACTED_SIZE_BYTES:
                        raise ArchiveLimitError("Фактический распакованный размер ZIP превышает лимит 1,5 ГБ")
                    target.write(chunk)


def unpack_archives(source_dir: str, output_dir: str | None = None):
    """
    Разархивирует все .rar файлы из source_dir в output_dir.
    Кросс-платформенно: работает на macOS и Linux.
    Битые архивы перемещаются в source_dir/_TROUBLES_/ с файлом _PROBLEM.txt.
    """
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir or source_dir).resolve()
    if not source_path.exists() or not source_path.is_dir():
        print(f"Директория с архивами {source_dir} не найдена.")
        return
    output_path.mkdir(parents=True, exist_ok=True)
    cleanup_stale_extracting_dirs(output_path)

    archive_files = sorted(
        (item for item in source_path.iterdir() if item.is_file() and item.suffix.lower() in {".rar", ".zip"}),
        key=lambda item: item.name.casefold(),
    )
    if not archive_files:
        print(f"В директории {source_dir} не найдено .rar или .zip файлов.")
        return

    # Внешний распаковщик нужен только для RAR. ZIP обрабатывает Python.
    has_rar = any(item.suffix.lower() == ".rar" for item in archive_files)
    tool = _find_extractor() if has_rar else None
    if has_rar and not tool:
        print(
            "❌ Не найден ни один инструмент для распаковки RAR.\n"
            "   macOS : brew install unar\n"
            "   Ubuntu: sudo apt install unar\n"
            "   или   : sudo apt install unrar\n"
            "   или   : sudo apt install p7zip-full\n"
            "   RAR будут оставлены во входной папке без изменений."
        )

    if tool:
        print(f"Инструмент распаковки RAR: {tool}")
    if any(item.suffix.lower() == ".zip" for item in archive_files):
        print("Инструмент распаковки ZIP: встроенный модуль Python")

    print(f"Найдено архивов для распаковки: {len(archive_files)}")

    for archive_file in archive_files:
        folder_name = archive_file.stem
        extract_dir = output_path / folder_name
        if (extract_dir / CONFLICT_MARKER).is_file():
            print(f"\n⏸ {archive_file.name}: конфликт ожидает решения оператора, архив оставлен на месте.")
            continue
        # Никогда не распаковываем поверх имеющейся папки: это может смешать
        # старые и новые файлы при повторном запуске.
        extract_dir_exists = extract_dir.exists()
        temp_dir = output_path / f".extracting_{folder_name}_{uuid.uuid4().hex}"

        print(f"\nРаспаковка {archive_file.name}")
        print(f"Во временную папку -> {temp_dir.name}/")

        error_reason = None

        try:
            archive_size = archive_file.stat().st_size
            if archive_size > MAX_ARCHIVE_SIZE_BYTES:
                raise ArchiveLimitError(
                    f"Архив превышает лимит 1,5 ГБ: {archive_size / 1_000_000_000:.2f} ГБ"
                )
            if archive_file.suffix.lower() == ".rar" and not tool:
                print("    ⏭ Пропуск: нет программы для распаковки RAR")
                continue
            if extract_dir_exists:
                raise FileExistsError(
                    f"Рабочая папка {extract_dir.name}/ уже существует; архив не будет перезаписан"
                )

            temp_dir.mkdir(parents=True, exist_ok=False)
            if archive_file.suffix.lower() == ".zip":
                _extract_zip_safely(archive_file, temp_dir)
            else:
                _preflight_rar(tool, archive_file)
                cmd = _build_command(tool, archive_file, temp_dir)
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=EXTRACT_TIMEOUT_SECONDS,
                )
                _reject_extracted_links(temp_dir)
            # Публикуем результат только после успешной распаковки.
            temp_dir.replace(extract_dir)
            print("Успешно распаковано.")

        except ArchiveLimitError as e:
            error_reason = str(e)
            print(f"❌ {error_reason}")

        except subprocess.CalledProcessError as e:
            error_msg = (
                e.stderr.decode("utf-8", errors="ignore").strip()
                or e.stdout.decode("utf-8", errors="ignore").strip()
                or f"{tool} вернул код ошибки {e.returncode}"
            )
            error_reason = f"Ошибка распаковки ({tool}): {error_msg}"
            print(f"❌ {error_reason}")

        except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
            error_reason = f"Ошибка безопасной проверки или распаковки: {e}"
            print(f"❌ {error_reason}")

        except subprocess.TimeoutExpired:
            error_reason = f"Распаковка превысила лимит {EXTRACT_TIMEOUT_SECONDS // 60} минут"
            print(f"❌ {error_reason}")

        except Exception as e:
            error_reason = f"Неожиданная ошибка: {e}"
            print(f"❌ {error_reason}")

        # При ошибке — переносим в _TROUBLES_
        if error_reason:
            troubles_dir = source_path / TROUBLES_DIR
            troubles_dir.mkdir(exist_ok=True)

            dest_archive = _unique_path(troubles_dir, archive_file.name)
            shutil.move(str(archive_file), str(dest_archive))

            # Сохраняем частичную распаковку для диагностики, не подмешивая её
            # в рабочую папку, которую затем обходит main.py.
            if temp_dir.exists():
                partial_dir = _unique_path(
                    troubles_dir, f"{archive_file.stem}_partial_{datetime.now().strftime('%H%M%S')}"
                )
                shutil.move(str(temp_dir), str(partial_dir))

            # Файл с причиной ошибки
            problem_file = troubles_dir / (archive_file.stem + "_PROBLEM.txt")
            problem_file.write_text(
                f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Архив: {archive_file.name}\n"
                f"Инструмент: {tool or 'zipfile'}\n"
                f"Причина: {error_reason}\n",
                encoding="utf-8",
            )
            print(f"    📁 Архив перемещён в {TROUBLES_DIR}/")


if __name__ == "__main__":
    target_directory = "test_archives"
    unpack_archives(target_directory)
