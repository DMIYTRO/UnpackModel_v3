"""Безопасная проверка LM Studio на инвентарном JSON без переименования файлов."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from classifier import classify_face_back, is_auxiliary_filename
from file_discovery import is_layout_file
from filename_parser import parse_filename
from local_llm import LocalLlmError, request_side_proposal
from website_parser import OrderDataError, SiteAccessError, fetch_suborders


def find_ambiguous_archives(payload: dict) -> list[dict]:
    """Выбирает двухсторонние записи, которые не решаются обычными правилами."""
    result = []
    for archive in payload.get("archives", []):
        archive_name = str(archive.get("archive_name", ""))
        if parse_filename(archive_name).get("sides") not in {"4-4", "5-5", "1-1"}:
            continue
        filenames = [
            name
            for name in archive.get("member_names", [])
            if (
                isinstance(name, str)
                and is_layout_file(Path(name))
                and not is_auxiliary_filename(Path(name).name)
            )
        ]
        if not filenames or len(filenames) % 2:
            continue
        ambiguous = False
        for index in range(0, len(filenames), 2):
            classified = classify_face_back(
                [Path(name).name for name in filenames[index : index + 2]]
            )
            if classified["face"] is None or classified["back"] is None:
                ambiguous = True
                break
        if ambiguous:
            result.append({"archive_name": archive_name, "filenames": filenames})
    return result


def expected_layout_count(order_ids: list[str], sides: str) -> int:
    """Для двухсторонней продукции считает по два файла на каждый заказ."""
    try:
        _, back_side = sides.split("-", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Некорректная сторонность: {sides}") from exc
    return len(order_ids) * (2 if int(back_side) > 0 else 1)


def run_json_test(
    input_path: Path,
    output_path: Path,
    *,
    limit: int = 5,
    fetch_orders=fetch_suborders,
) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    candidates = find_ambiguous_archives(payload)[:limit]
    report = {"source": str(input_path), "tested": []}

    print(f"Найдено спорных двухсторонних архивов: {len(candidates)}")
    for number, candidate in enumerate(candidates, start=1):
        archive_name = candidate["archive_name"]
        filenames = candidate["filenames"]
        parsed = parse_filename(archive_name)
        parsed_order = parsed.get("order_number")
        sides = parsed.get("sides")
        print(f"\n[{number}/{len(candidates)}] ❓ {archive_name}")
        print(f"    Файлы: {', '.join(filenames)}")
        item = {"archive_name": archive_name, "filenames": filenames}
        if not parsed_order or not sides:
            item.update(
                {
                    "status": "invalid_archive_name",
                    "reason": "Не удалось определить номер заказа или сторонность",
                }
            )
            print(f"    ⚠️ {item['reason']}")
            report["tested"].append(item)
            continue

        try:
            suborders = fetch_orders(parsed_order)
        except (SiteAccessError, OrderDataError) as exc:
            item.update({"status": "site_error", "reason": str(exc)})
            print(f"    ⚠️ Не удалось получить заказы с сайта: {exc}")
            report["tested"].append(item)
            continue

        order_ids = [parsed_order]
        order_ids.extend(value for value in suborders if value != parsed_order)
        extensions = sorted(
            {Path(filename).suffix.casefold() for filename in filenames}
        )
        has_pdf = ".pdf" in extensions
        two_sided = sides.split("-", maxsplit=1)[1] != "0"
        if has_pdf and two_sided:
            accepted_counts = list(
                range(len(order_ids), len(order_ids) * 2 + 1)
            )
        elif has_pdf:
            accepted_counts = [len(order_ids)]
        else:
            accepted_counts = [expected_layout_count(order_ids, sides)]
        expected = accepted_counts[0]
        files_per_order = expected // len(order_ids)
        item.update(
            {
                "sides": sides,
                "order_ids": order_ids,
                "site_order_count": len(order_ids),
                "actual_file_count": len(filenames),
                "accepted_file_counts": accepted_counts,
                "file_types": extensions,
            }
        )
        print(f"    🌐 Заказы с сайта ({len(order_ids)}): {', '.join(order_ids)}")
        if has_pdf and two_sided:
            print(
                f"    📊 С PDF: для {len(order_ids)} заказов допустимо от "
                f"{accepted_counts[0]} до {accepted_counts[-1]} файлов; "
                f"найдено {len(filenames)}"
            )
        else:
            print(
                f"    📊 Количество: {len(order_ids)} заказов × "
                f"{files_per_order} файл(а) = {expected}; найдено {len(filenames)}"
            )
        if len(filenames) not in accepted_counts:
            expected_text = " или ".join(str(value) for value in accepted_counts)
            item.update(
                {
                    "status": "count_mismatch",
                    "reason": (
                        f"Количество файлов не соответствует заказам: найдено "
                        f"{len(filenames)}, ожидалось {expected_text}"
                    ),
                }
            )
            print(f"    ⛔ {item['reason']}")
            print("    🤖 LM Studio не вызывается: сначала должно совпасть количество")
            report["tested"].append(item)
            continue
        if has_pdf:
            item.update(
                {
                    "status": "pdf_page_check_required",
                    "reason": (
                        "Количество файлов допустимо, но инвентарный JSON не содержит "
                        "число страниц PDF; нужна проверка распакованных макетов"
                    ),
                }
            )
            print(f"    ℹ️ {item['reason']}")
            print("    🤖 LM Studio не вызывается для определения числа страниц")
            report["tested"].append(item)
            continue

        print("    🤖 LM Studio: отправляю спорный случай")
        try:
            proposal = request_side_proposal(archive_name, filenames, order_ids)
            item.update(
                {
                    "status": "proposal",
                    "confidence": proposal.confidence,
                    "reason": proposal.reason,
                    "assignments": [
                        {
                            "order_id": assignment.order_id,
                            "face": assignment.face,
                            "back": assignment.back,
                        }
                        for assignment in proposal.assignments
                    ],
                }
            )
            print(f"    Уверенность: {proposal.confidence:.0%}")
            print(f"    Причина: {proposal.reason or 'не указана'}")
            for assignment in proposal.assignments:
                print(
                    f"    {assignment.order_id}: лицо={assignment.face}; "
                    f"оборот={assignment.back}"
                )
        except LocalLlmError as exc:
            item.update({"status": "ambiguous_or_error", "reason": str(exc)})
            print(f"    ⚠️ Нет безопасного предложения: {exc}")
        report["tested"].append(item)

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nРезультат записан: {output_path}")
    print("Файлы архивов не изменялись.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Проверить спорные имена из JSON через локальную LM Studio"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("tests/fixtures/archive_filename_inventory.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("llm_proposals.json"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    run_json_test(args.input, args.output, limit=max(args.limit, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
