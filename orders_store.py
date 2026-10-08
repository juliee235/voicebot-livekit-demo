"""Saves orders to a Google Sheet (when GOOGLE_SHEET_ID is set), otherwise to orders.jsonl.

New orders get status "กำลังทำ". Staff change it to "เสร็จแล้ว" in the sheet when the order
is ready, and the agent reads it back when the customer calls to check.
"""

import json
import os
from functools import cache
from pathlib import Path

COLUMNS = ["order_no", "created_at", "customer_name", "phone", "items", "total", "status"]
ORDERS_FILE = Path(__file__).parent / "orders.jsonl"


@cache
def _sheet():  # noqa: ANN202
    import gspread

    client = gspread.service_account(
        filename=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "google-service-account.json")
    )
    sheet = client.open_by_key(os.environ["GOOGLE_SHEET_ID"]).sheet1
    if not sheet.row_values(1):
        sheet.append_row(COLUMNS)
    return sheet


def _all_orders() -> list[dict]:
    if os.getenv("GOOGLE_SHEET_ID"):
        rows = _sheet().get_all_values()
        return [dict(zip(rows[0], row)) for row in rows[1:]]
    if not ORDERS_FILE.exists():
        return []
    return [json.loads(line) for line in ORDERS_FILE.read_text(encoding="utf-8").splitlines() if line]


def save_order(order: dict) -> int:
    """Save an order and return its order number. Blocking: call with asyncio.to_thread."""
    order = {**order, "order_no": len(_all_orders()) + 1, "status": "กำลังทำ"}
    if os.getenv("GOOGLE_SHEET_ID"):
        _sheet().append_row([order[column] for column in COLUMNS])  # RAW: phone keeps its leading 0
    else:
        with ORDERS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(order, ensure_ascii=False) + "\n")
    return order["order_no"]


def latest_order(phone: str) -> dict | None:
    """The newest order for this phone number. Blocking: call with asyncio.to_thread."""
    return next((o for o in reversed(_all_orders()) if str(o["phone"]) == phone), None)
