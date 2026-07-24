"""Подсчёт печатных страниц в PDF-макетах."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfPageCountError(RuntimeError):
    """Количество страниц PDF нельзя безопасно определить."""


def count_pdf_pages(path: str | Path) -> int:
    """Возвращает фактическое количество страниц PDF."""
    pdf_path = Path(path)
    if pdf_path.suffix.casefold() != ".pdf":
        raise PdfPageCountError(f"Файл не является PDF: {pdf_path}")
    try:
        page_count = len(PdfReader(str(pdf_path), strict=False).pages)
    except (OSError, PdfReadError, ValueError) as exc:
        raise PdfPageCountError(f"Не удалось прочитать PDF {pdf_path.name}: {exc}") from exc
    if page_count < 1:
        raise PdfPageCountError(f"PDF не содержит страниц: {pdf_path.name}")
    return page_count


def count_layout_units(files: list[Path]) -> tuple[int, dict[Path, int]]:
    """Считает стороны: обычный файл = 1, PDF = число его страниц."""
    details = {}
    total = 0
    for path in files:
        units = count_pdf_pages(path) if path.suffix.casefold() == ".pdf" else 1
        details[path] = units
        total += units
    return total, details
