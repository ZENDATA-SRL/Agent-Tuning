from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

_TOOLS_PATH = Path(__file__).resolve().parent.parent / "examples" / "tools.json"
TOOLS: list[dict] = json.loads(_TOOLS_PATH.read_text(encoding="utf-8"))

_BOOKINGS: list[dict] = []

_SHOP = {
    "name": "Barbiera",
    "address": "Via Andrea Appiani 5, 23842 Bosisio Parini, LC, Italia",
    "phone": "393518743543",
    "hours": {"lun-ven": "09:00-20:00", "sab": "08:30-13:00", "dom": "chiuso"},
    "team": ["Giorgia"],
    "services": [
        {"name": "Taglio capelli", "duration_min": 30, "price_eur": 21.0},
        {"name": "Taglio e barba", "duration_min": 45, "price_eur": 31.0},
        {"name": "Barba", "duration_min": 15, "price_eur": 10.0},
        {"name": "Piega", "duration_min": 30, "price_eur": 20.0},
        {"name": "Taglio bambino", "duration_min": 15, "price_eur": 16.0},
    ],
}

_DEFAULT_SLOTS = ["09:30", "10:00", "11:00", "15:00", "16:30", "17:30", "18:00"]


def _dates_between(start: str, end: str) -> list[str]:
    from datetime import datetime, timedelta

    fmt = "%d-%m-%Y"
    a = datetime.strptime(start, fmt)
    b = datetime.strptime(end, fmt)
    if b < a:
        a, b = b, a
    out = []
    cur = a
    while cur <= b and len(out) < 14:
        if cur.weekday() != 6:
            out.append(cur.strftime(fmt))
        cur += timedelta(days=1)
    return out


def get_shop_info(**_: Any) -> dict:
    return _SHOP


def get_service_availability(
    service: str,
    start_date_str: str,
    end_date_str: str,
    professional: str | None = None,
) -> dict:
    pro = professional or "Giorgia"
    days = []
    for date_str in _dates_between(start_date_str, end_date_str):
        taken = {
            b["time"]
            for b in _BOOKINGS
            if b["date"] == date_str and b["professional"] == pro and b["service"] == service
        }
        slots = [s for s in _DEFAULT_SLOTS if s not in taken]
        if slots:
            days.append({
                "date": date_str,
                "by_professional": [{"professional": pro, "slots": slots}],
            })
    return {"service": service, "available": days}


def get_professional_availability(
    start_date_str: str,
    end_date_str: str,
    professional: str | None = None,
) -> dict:
    pros = [professional] if professional else ["Giorgia"]
    blocks = []
    for date_str in _dates_between(start_date_str, end_date_str):
        for pro in pros:
            blocks.append({
                "date": date_str,
                "professional": pro,
                "blocks": ["09:00-12:00 (3h)", "15:00-19:00 (4h)"],
            })
    return {"blocks": blocks}


def book_appointment(
    service: str,
    date_str: str,
    time_str: str,
    professional: str | None = None,
    notes: str | None = None,
    allow_concurrent_booking: bool = False,
) -> dict:
    pro = professional or "Giorgia"
    conflict = any(
        b["date"] == date_str and b["time"] == time_str and b["professional"] == pro
        for b in _BOOKINGS
    )
    if conflict and not allow_concurrent_booking:
        return {
            "status": "error",
            "message": f"Slot {date_str} {time_str} with {pro} is unavailable.",
        }
    booking = {
        "status": "confirmed",
        "booking_id": str(uuid.uuid4()),
        "service": service,
        "professional": pro,
        "date": date_str,
        "time": time_str,
        "notes": notes,
    }
    _BOOKINGS.append(booking)
    return booking


def cancel_appointment(
    service: str,
    date_str: str,
    time_str: str,
    professional: str,
) -> dict:
    for i, b in enumerate(_BOOKINGS):
        if (
            b["service"] == service
            and b["date"] == date_str
            and b["time"] == time_str
            and b["professional"] == professional
        ):
            removed = _BOOKINGS.pop(i)
            return {"status": "cancelled", **removed}
    return {"status": "error", "message": "Booking not found."}


def modify_appointment(
    original_service: str,
    original_date: str,
    original_time: str,
    original_professional: str,
    new_service: str,
    new_date: str,
    new_time: str,
    new_professional: str,
) -> dict:
    cancel = cancel_appointment(
        original_service, original_date, original_time, original_professional
    )
    if cancel.get("status") == "error":
        return cancel
    return book_appointment(
        service=new_service,
        date_str=new_date,
        time_str=new_time,
        professional=new_professional,
    )


_HANDLERS = {
    "get_shop_info": get_shop_info,
    "get_service_availability": get_service_availability,
    "get_professional_availability": get_professional_availability,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "modify_appointment": modify_appointment,
}


def run_tool(name: str, arguments: dict | str) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "message": "Invalid JSON arguments"})
    fn = _HANDLERS.get(name)
    if fn is None:
        return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
    try:
        result = fn(**(arguments or {}))
    except TypeError as exc:
        return json.dumps({"status": "error", "message": str(exc)})
    return json.dumps(result, ensure_ascii=False)
