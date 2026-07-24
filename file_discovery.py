"""Поиск файлов макета внутри распакованного заказа."""
from pathlib import Path

from classifier import is_auxiliary_filename

# Форматы, которые могут быть печатными макетами. Превью, описания заказа и
# прочие служебные файлы не должны влиять на количество сторон.
SUPPORTED_LAYOUT_EXTENSIONS = frozenset({
    ".ai",
    ".cdr",
    ".eps",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".ps",
    ".psb",
    ".psd",
    ".svg",
    ".tif",
    ".tiff",
})


def is_layout_file(path: Path) -> bool:
    """Проверяет, относится ли файл к поддерживаемому формату макета."""
    return path.suffix.casefold() in SUPPORTED_LAYOUT_EXTENSIONS


def list_layout_files(root: Path) -> list[Path]:
    """Возвращает поддерживаемые файлы макетов в папке и подпапках.

    Служебные файлы и папки, начинающиеся с точки, исключаются: в частности,
    маркер ``.done`` не считается макетом при повторном запуске. Файлы иных
    форматов (например, TXT/JSON или изображения-превью) также исключаются.
    """
    files = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not is_layout_file(candidate):
            continue
        if is_auxiliary_filename(candidate.name):
            continue
        files.append(candidate)

    return sorted(files, key=lambda item: str(item.relative_to(root)).casefold())
