"""
Adds a `tools` field to every record in the traces JSONL dataset and
writes a standalone tools definition file.

Run:
    python scripts/add_tools_to_dataset.py
"""
import json
from pathlib import Path

# ── Tool definitions (OpenAI function-calling schema) ──────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_shop_info",
            "description": (
                "Get shop details, team bios, working hours, and service descriptions.\n"
                "Use for \"about us\", \"who works here\", \"what are your hours\", "
                "or \"what does X include\" questions."
            ),
            "parameters": {
                "properties": {},
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_availability",
            "description": (
                "Find all bookable time slots for a specific service.\n\n"
                "Use to:\n"
                "- Show customers available options before booking\n"
                "- Find alternatives when a requested slot is busy\n"
                "- Answer \"when can I book X?\" questions\n\n"
                "Args:\n"
                "    service: Exact service name\n"
                "    start_date_str: Start date (DD-MM-YYYY)\n"
                "    end_date_str: End date (DD-MM-YYYY)\n"
                "    professional: Optional - filter by professional name\n\n"
                "Returns actual bookable start times grouped by date and professional."
            ),
            "parameters": {
                "properties": {
                    "service": {"type": "string"},
                    "start_date_str": {"type": "string"},
                    "end_date_str": {"type": "string"},
                    "professional": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                },
                "required": ["service", "start_date_str", "end_date_str"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_professional_availability",
            "description": (
                "Get free time blocks for professionals.\n"
                "Use when customer wants to know general availability.\n\n"
                "Args:\n"
                "    start_date_str: Start date (DD-MM-YYYY)\n"
                "    end_date_str: End date (DD-MM-YYYY)\n"
                "    professional: Optional - filter by name, otherwise returns all\n\n"
                "Returns free blocks with duration (e.g., \"09:00-12:00 (3h)\")."
            ),
            "parameters": {
                "properties": {
                    "start_date_str": {"type": "string"},
                    "end_date_str": {"type": "string"},
                    "professional": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                },
                "required": ["start_date_str", "end_date_str"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Create a new appointment for the customer.\n\n"
                "Args:\n"
                "    service: Exact service name\n"
                "    date_str: Date (DD-MM-YYYY)\n"
                "    time_str: Start time (HH:MM, rounded to nearest 15 min)\n"
                "    professional: Professional's exact name "
                "(optional - if omitted, assigns first available)\n"
                "    notes: Optional notes from the customer\n"
                "    allow_concurrent_booking: Set to True ONLY if the customer explicitly "
                "mentions booking for another person or multiple people for the same time.\n\n"
                "Returns confirmation with booking details, or error if slot unavailable."
            ),
            "parameters": {
                "properties": {
                    "service": {"type": "string"},
                    "date_str": {"type": "string"},
                    "time_str": {"type": "string"},
                    "professional": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "notes": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                    "allow_concurrent_booking": {
                        "default": False,
                        "type": "boolean",
                    },
                },
                "required": ["service", "date_str", "time_str"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": (
                "Cancel an existing appointment.\n\n"
                "Use exact details from the customer's booking.\n"
                "If multiple bookings could match, ask customer to clarify.\n\n"
                "Args:\n"
                "    service: Service name\n"
                "    date_str: Date (DD-MM-YYYY)\n"
                "    time_str: Time (HH:MM)\n"
                "    professional: Professional's name\n\n"
                "Returns cancellation confirmation or error if not found."
            ),
            "parameters": {
                "properties": {
                    "service": {"type": "string"},
                    "date_str": {"type": "string"},
                    "time_str": {"type": "string"},
                    "professional": {"type": "string"},
                },
                "required": ["service", "date_str", "time_str", "professional"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_appointment",
            "description": (
                "Change an existing appointment to a new time, date, service, or professional.\n\n"
                "Args:\n"
                "    original_service: Current service name\n"
                "    original_date: Current date in DD-MM-YYYY\n"
                "    original_time: Current time in HH:MM\n"
                "    original_professional: Current professional\n"
                "    new_service: New service name\n"
                "    new_date: New date in DD-MM-YYYY\n"
                "    new_time: New time in HH:MM\n"
                "    new_professional: New professional\n\n"
                "Returns success with new details, or explains why the change couldn't be made."
            ),
            "parameters": {
                "properties": {
                    "original_service": {"type": "string"},
                    "original_date": {"type": "string"},
                    "original_time": {"type": "string"},
                    "original_professional": {"type": "string"},
                    "new_service": {"type": "string"},
                    "new_date": {"type": "string"},
                    "new_time": {"type": "string"},
                    "new_professional": {"type": "string"},
                },
                "required": [
                    "original_service",
                    "original_date",
                    "original_time",
                    "original_professional",
                    "new_service",
                    "new_date",
                    "new_time",
                    "new_professional",
                ],
                "type": "object",
            },
        },
    },
]


def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    dataset_path = data_dir / "traces-2026-05-18.jsonl"
    tools_path = data_dir / "tools.json"

    # ── Write standalone tools definition file ────────────────────────────
    tools_path.write_text(json.dumps(TOOLS, indent=2, ensure_ascii=False))
    print(f"Wrote {len(TOOLS)} tool definitions → {tools_path}")

    # ── Patch dataset in-place ─────────────────────────────────────────────
    records = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    already_has_tools = sum(1 for r in records if "tools" in r)
    if already_has_tools:
        print(f"  {already_has_tools} records already had a 'tools' field — overwriting.")

    updated = []
    for rec in records:
        updated.append({
            "trace_id": rec["trace_id"],
            "full_trace": rec["full_trace"],
            "final_answer": rec["final_answer"],
            "tools": TOOLS,
        })

    with open(dataset_path, "w", encoding="utf-8") as fh:
        for rec in updated:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Updated {len(updated)} records in {dataset_path}")
    print("New schema: trace_id | full_trace | final_answer | tools")


if __name__ == "__main__":
    main()
