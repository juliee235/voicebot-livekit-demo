"""Tools the agent can call: place a pickup order, check an order's status, end the call."""

import asyncio
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from livekit.agents import function_tool, llm
from livekit.agents.beta.tools import EndCallTool
from pydantic import BaseModel, Field

from menu import MENU, PICKUP_MINUTES, PROMOTIONS
from orders_store import latest_order, save_order

BANGKOK = ZoneInfo("Asia/Bangkok")


class OrderItem(BaseModel):
    menu: Literal[tuple(MENU)]  # type: ignore[valid-type]  # only names from menu.py
    quantity: int
    special: bool = Field(description="True for the พิเศษ size, False for ธรรมดา")
    note: str = Field(description="Requests like ไม่ใส่ผัก or แยกน้ำ. Empty if none. Not the size.")


def discount_for(subtotal: int) -> int:
    weekday = datetime.now(BANGKOK).weekday()
    return max(
        (
            p.get("discount", 0) + subtotal * p.get("percent", 0) // 100
            for p in PROMOTIONS
            if subtotal >= p["min_total"] and p.get("weekday", weekday) == weekday
        ),
        default=0,
    )


@function_tool
async def place_order(customer_name: str, phone: str, items: list[OrderItem]) -> str:
    """Place a pickup order. Call it once, after reading the items back to the customer
    and the customer has said yes.

    Args:
        customer_name: The customer's name.
        phone: The customer's phone number, digits only.
        items: Everything the customer ordered.
    """
    phone = "".join(c for c in phone if c.isdigit())
    if len(phone) not in (9, 10):
        return "เบอร์โทรไม่ครบ ให้ขอเบอร์โทรอีกครั้ง"

    subtotal = 0
    lines = []
    for item in items:
        prices = MENU[item.menu]
        special = item.special and "special" in prices
        subtotal += prices["special" if special else "price"] * item.quantity
        lines.append(f"{item.menu}{'พิเศษ' if special else ''} {item.quantity}" + (f" ({item.note})" if item.note else ""))
    discount = discount_for(subtotal)
    total = subtotal - discount

    order_no = await asyncio.to_thread(
        save_order,
        {
            "created_at": datetime.now(BANGKOK).strftime("%Y-%m-%d %H:%M"),
            "customer_name": customer_name,
            "phone": phone,
            "items": ", ".join(lines),
            "total": total,
        },
    )
    promo = f" (ลดจากโปร {discount} บาทแล้ว)" if discount else ""
    return f"ออเดอร์ที่ {order_no} ยอดชำระ {total} บาท{promo} มารับที่ร้านได้ในอีกประมาณ {PICKUP_MINUTES} นาที"


@function_tool
async def check_order_status(phone: str) -> str:
    """Check whether the customer's latest order is ready, by the phone number they ordered with.

    Args:
        phone: The phone number used for the order, digits only.
    """
    order = await asyncio.to_thread(latest_order, "".join(c for c in phone if c.isdigit()))
    if order is None:
        return "ไม่พบออเดอร์ของเบอร์นี้ ให้ทวนเบอร์กับลูกค้าอีกครั้ง"
    return f"ออเดอร์ที่ {order['order_no']} ({order['items']}) สถานะ: {order['status']}"


def build_tools() -> list[llm.Tool | llm.Toolset]:
    """Fresh tools for each call (EndCallTool keeps per-call state)."""
    return [
        place_order,
        check_order_status,
        EndCallTool(end_instructions="กล่าวลาผู้ใช้สั้นๆ เป็นภาษาไทย", ignore_on_enter=True),
    ]
