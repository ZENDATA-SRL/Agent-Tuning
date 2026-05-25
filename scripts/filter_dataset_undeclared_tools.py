"""Filter out training records that reference tools not declared in tools.json.

Background:
  The exported `traces-2026-05-18.jsonl` contains 7 traces that call
  6 distinct `owner_*` tools (e.g. `owner_get_bookings_revenue_overview`)
  which are NOT in `tools.json`. Training on these would teach the model
  to hallucinate tool calls — the assistant turn shows a tool call whose
  schema the model has never seen in `tools=[...]`. Worse, the corresponding
  `role:"tool"` messages would be rendered as `<|tool_response>response:?...`
  with no anchoring declaration.

  We move them into a quarantine file (`error_traces.jsonl`) for later
  inspection rather than deleting them outright, in case the owner-facing
  tools become part of the supported catalog later.

  All other 1316 traces remain untouched, including the 517 records
  without any tool call (kept on purpose: the model learns *when not* to
  call a tool, which feeds directly into the compliance reward).

Usage:
    python scripts/filter_dataset_undeclared_tools.py

Idempotent: safe to re-run; reads/writes the same paths.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "examples" / "traces-2026-05-18.jsonl"
TOOLS = ROOT / "examples" / "tools.json"
ERROR_OUT = ROOT / "examples" / "error_traces.jsonl"


def _record_tool_names(record: dict) -> set[str]:
    """Collect every tool name invoked anywhere in `record["full_trace"]`."""
    names: set[str] = set()
    for msg in record.get("full_trace", []):
        for tc in (msg.get("tool_calls") or []):
            name = tc.get("name") or tc.get("function", {}).get("name", "")
            if name:
                names.add(name)
    return names


def main() -> None:
    declared = {
        t["function"]["name"]
        for t in json.loads(TOOLS.read_text(encoding="utf-8"))
    }
    print(f"Declared tools ({len(declared)}): {sorted(declared)}")

    records: list[dict] = []
    with DATASET.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    total = len(records)

    keep: list[dict] = []
    drop: list[dict] = []
    for rec in records:
        used = _record_tool_names(rec)
        undeclared = used - declared
        if undeclared:
            rec["_undeclared_tools"] = sorted(undeclared)
            drop.append(rec)
        else:
            keep.append(rec)

    if not drop:
        print(f"Nothing to filter: all {total} records use only declared tools.")
        return

    print(f"Filtering: {len(drop)} dropped, {len(keep)} kept (out of {total}).")
    print("Undeclared tool occurrences in dropped records:")
    counts: dict[str, int] = {}
    for rec in drop:
        for name in rec["_undeclared_tools"]:
            counts[name] = counts.get(name, 0) + 1
    for name in sorted(counts):
        print(f"  {name:40s} {counts[name]}")

    with ERROR_OUT.open("w", encoding="utf-8") as fh:
        for rec in drop:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote quarantine file → {ERROR_OUT} ({len(drop)} records)")

    with DATASET.open("w", encoding="utf-8") as fh:
        for rec in keep:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Rewrote dataset       → {DATASET} ({len(keep)} records)")


if __name__ == "__main__":
    main()
