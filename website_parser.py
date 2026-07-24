"""Клиент API sborka.ua для получения подзаказов."""

from __future__ import annotations

import os
import re

import requests
from dotenv import load_dotenv


API_URL = "https://sborka.ua/api.php"


class WebsiteParserError(RuntimeError):
    """Базовая ошибка получения данных о заказе."""


class SiteAccessError(WebsiteParserError):
    """API недоступен, отклонил запрос или вернул ошибку HTTP."""


class OrderDataError(WebsiteParserError):
    """Номер заказа или ответ API имеет неожиданный формат."""


def _validate_order_number(order_number: str) -> str:
    value = str(order_number).strip()
    if not value.isdigit():
        raise OrderDataError(f"Некорректный номер заказа: {order_number!r}")
    return value


def parse_suborders_response(response_text: str) -> list[str]:
    """Разбирает текстовый ответ API: номера, разделённые запятыми."""
    value = response_text.strip().rstrip(",").strip()
    if not value:
        return []

    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if not re.fullmatch(r"\d+", item)]
    if invalid:
        preview = response_text.strip().replace("\n", " ")[:160]
        raise OrderDataError(f"API вернул неожиданный ответ: {preview!r}")

    # Убираем дубликаты и сохраняем числовой порядок подзаказов.
    return sorted(set(items), key=int)


def fetch_suborders(
    order_number: str,
    api_key: str | None = None,
    timeout: int = 10,
) -> list[str]:
    """Получает подзаказы через единственный запрос ``getSubOrders``."""
    order_number = _validate_order_number(order_number)
    load_dotenv()
    key = api_key or os.getenv("SBORKA_API_KEY")
    if not key:
        raise SiteAccessError("В .env не задан SBORKA_API_KEY")

    print(f"    🌐 Сайт: отправляю запрос getSubOrders для заказа {order_number}")
    try:
        response = requests.post(
            API_URL,
            params={"action": "getSubOrders", "id": order_number},
            data={"api_key": key},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SiteAccessError(f"Ошибка запроса к API sborka.ua: {exc}") from exc

    suborders = parse_suborders_response(response.text)
    shown = ", ".join(suborders) if suborders else "нет"
    print(f"    🌐 Сайт: получены подзаказы: {shown}")
    return suborders


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Получить подзаказы из API sborka.ua")
    parser.add_argument("order_number", help="Номер основного заказа")
    args = parser.parse_args()
    print("Найдены подзаказы:", fetch_suborders(args.order_number))
