"""Generate a realistic agentic-trace dataset for SFT smoke / dev runs.

The real production dataset (`data/traces-2026-05-18.jsonl`) is exported
from Langfuse and patched in place by `scripts/add_tools_to_dataset.py`
with the 6-tool catalog of an Italian beauty-salon booking agent. We
mirror that exact wire format here so a model trained on this synthetic
file can be drop-in compared against the real one:

    {
      "trace_id":     str,
      "full_trace":   list[ {role, content, [tool_calls], [tool_call_id]} ],
      "final_answer": str,
      "tools":        list[OpenAI function-schema],
    }

Output: examples/realistic_dataset.jsonl (50 records, deterministic seed).

Distribution (by design, see SCENARIO_PLAN below):
  ~30%  single tool-call   (info, availability, cancel)
  ~50%  multi  tool-call   (check+book, modify, cancel-and-rebook, ...)
  ~15%  clarification      (ambiguous service / date / professional)
  ~ 5%  edge case          (slot taken, booking not found, tool error)

The generator is deterministic (`random.seed(SEED)`) so re-running it
produces a byte-identical file — useful for CI smoke tests.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.add_tools_to_dataset import TOOLS  # noqa: E402

SEED = 20260524
OUTPUT_PATH = ROOT / "examples" / "realistic_dataset.jsonl"
TARGET_TRACES = 50

SYSTEM_PROMPT = (
    "Sei l'assistente virtuale di Bellezza Studio, un salone di bellezza a Milano. "
    "Aiuti i clienti a trovare informazioni sui servizi, controllare disponibilità, "
    "prenotare, modificare e cancellare appuntamenti usando gli strumenti a tua "
    "disposizione. Rispondi sempre in italiano, in modo professionale ma amichevole. "
    "Le date sono nel formato DD-MM-YYYY e gli orari nel formato HH:MM. Se la "
    "richiesta è ambigua chiedi un chiarimento prima di chiamare uno strumento. "
    "Non inventare mai disponibilità o conferme che non provengono dagli strumenti."
)

SERVICES: tuple[str, ...] = (
    "Taglio Donna",
    "Taglio Uomo",
    "Piega",
    "Colore",
    "Manicure",
    "Pedicure",
    "Massaggio Rilassante",
    "Pulizia Viso",
    "Trattamento Anti-età",
    "Ceretta Gambe",
)

PROFESSIONALS: tuple[str, ...] = (
    "Giulia",
    "Marco",
    "Sofia",
    "Luca",
    "Chiara",
    "Alessia",
)

SERVICE_TO_PROS: dict[str, tuple[str, ...]] = {
    "Taglio Donna":            ("Giulia", "Sofia", "Chiara"),
    "Taglio Uomo":             ("Marco", "Luca"),
    "Piega":                   ("Giulia", "Sofia", "Chiara"),
    "Colore":                  ("Giulia", "Chiara"),
    "Manicure":                ("Sofia", "Alessia"),
    "Pedicure":                ("Sofia", "Alessia"),
    "Massaggio Rilassante":    ("Alessia",),
    "Pulizia Viso":            ("Chiara", "Alessia"),
    "Trattamento Anti-età":    ("Chiara",),
    "Ceretta Gambe":           ("Sofia", "Alessia"),
}


# ── Trace-building primitives ─────────────────────────────────────────────


@dataclass
class TraceBuilder:
    """Accumulator for a single trace.

    Provides one method per role so the per-scenario code reads as a
    transcript: `b.user(...); b.tool_call(...); b.tool_result(...); b.ai(...)`.
    `tool_call_id`s are auto-generated and threaded through automatically
    so that the assistant's `tool_calls[*].id` match the corresponding
    `role:"tool"` message's `tool_call_id`.
    """

    trace_id: str
    messages: list[dict] = field(default_factory=list)
    final_answer: str = ""
    _next_call_id: int = 1
    _last_call_ids: list[str] = field(default_factory=list)

    def system(self, content: str = SYSTEM_PROMPT) -> None:
        self.messages.append({"role": "system", "content": content})

    def user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def ai(self, content: str, *, is_final: bool = False) -> None:
        self.messages.append({"role": "assistant", "content": content})
        if is_final:
            self.final_answer = content

    def tool_calls(self, calls: list[tuple[str, dict]]) -> None:
        """Append an assistant turn that invokes one or more tools.

        `calls` is a list of `(tool_name, args_dict)` tuples. The per-call
        ids are stashed in `_last_call_ids` so the next `tool_result(s)`
        invocations can reference them in the right order.
        """
        formatted = []
        self._last_call_ids = []
        for name, args in calls:
            cid = f"call_{self._next_call_id}"
            self._next_call_id += 1
            self._last_call_ids.append(cid)
            formatted.append({
                "id": cid,
                "name": name,
                "args": args,
            })
        self.messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": formatted,
        })

    def tool_result(self, content: str | dict, *, slot: int = 0) -> None:
        """Append a `role:"tool"` message bound to the slot-th most-recent
        tool_call id."""
        if not self._last_call_ids:
            raise RuntimeError(f"tool_result without a preceding tool_calls in {self.trace_id}")
        cid = self._last_call_ids[slot]
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        self.messages.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": content,
        })

    def build(self) -> dict:
        if not self.final_answer:
            raise RuntimeError(f"Trace {self.trace_id} has no final answer")
        return {
            "trace_id": self.trace_id,
            "full_trace": self.messages,
            "final_answer": self.final_answer,
            "tools": TOOLS,
        }


# ── Domain helpers ────────────────────────────────────────────────────────


def _date_str(rng: random.Random) -> str:
    """Random business-future date in DD-MM-YYYY (next ~60 days)."""
    day = rng.randint(1, 28)
    month = rng.randint(6, 12)
    return f"{day:02d}-{month:02d}-2026"


def _date_range(rng: random.Random, span_days: int = 7) -> tuple[str, str]:
    start_day = rng.randint(1, 21)
    month = rng.randint(6, 12)
    end_day = start_day + rng.randint(2, span_days)
    return (f"{start_day:02d}-{month:02d}-2026", f"{end_day:02d}-{month:02d}-2026")


def _time_slot(rng: random.Random) -> str:
    hour = rng.choice([9, 10, 11, 14, 15, 16, 17, 18])
    minute = rng.choice([0, 15, 30, 45])
    return f"{hour:02d}:{minute:02d}"


def _availability_payload(
    service: str,
    start: str,
    end: str,
    rng: random.Random,
    *,
    professional: str | None = None,
) -> dict:
    """Synthetic but realistic-looking availability response."""
    pros = (professional,) if professional else SERVICE_TO_PROS[service]
    days = []
    cursor_day = int(start[:2])
    span = max(1, int(end[:2]) - cursor_day + 1)
    for d in range(min(span, 4)):
        date_str = f"{cursor_day + d:02d}-{start[3:]}"
        slots_for_day = []
        for pro in pros:
            n_slots = rng.randint(1, 3)
            slots = sorted({_time_slot(rng) for _ in range(n_slots)})
            slots_for_day.append({"professional": pro, "slots": slots})
        days.append({"date": date_str, "by_professional": slots_for_day})
    return {"service": service, "available": days}


def _booking_confirmation(
    service: str,
    date_str: str,
    time_str: str,
    professional: str,
    rng: random.Random,
) -> dict:
    return {
        "status": "confirmed",
        "booking_id": f"BK-{rng.randint(10000, 99999)}",
        "service": service,
        "date": date_str,
        "time": time_str,
        "professional": professional,
    }


# ── Scenario generators ───────────────────────────────────────────────────
# Each `gen_*` returns a finished dict (via TraceBuilder.build()). Generators
# are deterministic given an `rng`. Categories below match SCENARIO_PLAN.


# ----- single-tool scenarios -----

def gen_single_shop_info(rng: random.Random, idx: int) -> dict:
    """User asks for shop info / hours / team → one get_shop_info call."""
    questions = [
        "Ciao, a che ora aprite domani?",
        "Buongiorno, mi sapete dire chi lavora da voi?",
        "Quali sono i vostri orari di apertura?",
        "Vorrei sapere qualcosa sul vostro salone.",
        "Mi potete dare un po' di informazioni su Bellezza Studio?",
    ]
    q = questions[idx % len(questions)]
    info = {
        "name": "Bellezza Studio",
        "address": "Via Torino 42, Milano",
        "hours": {"lun-ven": "09:00-19:00", "sab": "09:00-17:00", "dom": "chiuso"},
        "team": list(PROFESSIONALS),
        "phone": "+39 02 1234567",
    }
    final_map = {
        0: "Apriamo alle 09:00 e chiudiamo alle 19:00 dal lunedì al venerdì. Il sabato siamo aperti 09:00-17:00, mentre la domenica siamo chiusi.",
        1: "Da Bellezza Studio lavorano Giulia, Marco, Sofia, Luca, Chiara e Alessia. Vuoi sapere di più su qualcuno in particolare?",
        2: "I nostri orari sono: lun-ven 09:00-19:00, sabato 09:00-17:00, domenica chiuso.",
        3: "Bellezza Studio è in Via Torino 42 a Milano. Il nostro team è composto da 6 professionisti e siamo aperti tutta la settimana tranne la domenica. Posso aiutarti con qualcosa?",
        4: "Bellezza Studio si trova in Via Torino 42 a Milano, telefono +39 02 1234567. Siamo aperti dal lunedì al sabato. Vuoi prenotare un servizio?",
    }
    b = TraceBuilder(trace_id=f"realistic-shop-info-{idx:02d}")
    b.system()
    b.user(q)
    b.tool_calls([("get_shop_info", {})])
    b.tool_result(info)
    b.ai(final_map[idx % len(questions)], is_final=True)
    return b.build()


def gen_single_service_availability(rng: random.Random, idx: int) -> dict:
    """User asks 'when can I book X' → one get_service_availability call."""
    service = rng.choice(SERVICES)
    start, end = _date_range(rng, span_days=5)
    b = TraceBuilder(trace_id=f"realistic-service-avail-{idx:02d}")
    b.system()
    b.user(f"Quando posso prenotare un {service.lower()} tra il {start} e il {end}?")
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": start,
        "end_date_str": end,
    })])
    payload = _availability_payload(service, start, end, rng)
    b.tool_result(payload)
    first_day = payload["available"][0]
    pro = first_day["by_professional"][0]
    b.ai(
        f"Per {service} tra il {start} e il {end} ci sono questi slot disponibili: "
        f"il {first_day['date']} con {pro['professional']} alle {', '.join(pro['slots'])}. "
        f"Vuoi che ti prenoti uno di questi orari?",
        is_final=True,
    )
    return b.build()


def gen_single_pro_availability(rng: random.Random, idx: int) -> dict:
    """User asks general 'when is X free' → get_professional_availability."""
    pro = rng.choice(PROFESSIONALS)
    start, end = _date_range(rng, span_days=3)
    b = TraceBuilder(trace_id=f"realistic-pro-avail-{idx:02d}")
    b.system()
    b.user(f"Quando è libera {pro} la prossima settimana?")
    b.tool_calls([("get_professional_availability", {
        "start_date_str": start,
        "end_date_str": end,
        "professional": pro,
    })])
    blocks = [
        {"date": start, "professional": pro, "free": ["09:00-11:00 (2h)", "15:00-17:00 (2h)"]},
        {"date": end,   "professional": pro, "free": ["10:00-12:00 (2h)"]},
    ]
    b.tool_result({"blocks": blocks})
    b.ai(
        f"{pro} è libera il {start} dalle 09:00 alle 11:00 e dalle 15:00 alle 17:00, "
        f"e il {end} dalle 10:00 alle 12:00. Per quale servizio vorresti prenotare?",
        is_final=True,
    )
    return b.build()


def gen_single_cancel(rng: random.Random, idx: int) -> dict:
    """User has all booking details, agent cancels in one shot."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    date_str = _date_str(rng)
    time_str = _time_slot(rng)
    b = TraceBuilder(trace_id=f"realistic-single-cancel-{idx:02d}")
    b.system()
    b.user(
        f"Devo cancellare la mia prenotazione di {service.lower()} con {pro} "
        f"il {date_str} alle {time_str}."
    )
    b.tool_calls([("cancel_appointment", {
        "service": service,
        "date_str": date_str,
        "time_str": time_str,
        "professional": pro,
    })])
    b.tool_result({"status": "cancelled", "service": service, "date": date_str, "time": time_str})
    b.ai(
        f"Ho cancellato la tua prenotazione di {service.lower()} con {pro} "
        f"il {date_str} alle {time_str}. Ci dispiace che tu non possa venire — "
        f"se vuoi riprenotare per un altro giorno, fammi sapere!",
        is_final=True,
    )
    return b.build()


# ----- multi-tool scenarios -----

def gen_check_then_book(rng: random.Random, idx: int) -> dict:
    """Classic 2-call flow: check availability, present options, then book
    after the user picks one."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    start, end = _date_range(rng, span_days=4)
    chosen_date = start
    chosen_time = _time_slot(rng)

    b = TraceBuilder(trace_id=f"realistic-check-book-{idx:02d}")
    b.system()
    b.user(f"Vorrei prenotare un {service.lower()} con {pro} la prossima settimana.")
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": start,
        "end_date_str": end,
        "professional": pro,
    })])
    b.tool_result(_availability_payload(service, start, end, rng, professional=pro))
    b.ai(
        f"Con {pro} ho trovato disponibilità il {chosen_date} alle {chosen_time}, "
        f"e qualche altro slot nei giorni successivi. Vuoi che prenoti il {chosen_date} "
        f"alle {chosen_time}?"
    )
    b.user("Sì, perfetto, prenota.")
    b.tool_calls([("book_appointment", {
        "service": service,
        "date_str": chosen_date,
        "time_str": chosen_time,
        "professional": pro,
    })])
    b.tool_result(_booking_confirmation(service, chosen_date, chosen_time, pro, rng))
    b.ai(
        f"Fatto! Ho prenotato {service.lower()} con {pro} il {chosen_date} "
        f"alle {chosen_time}. A presto!",
        is_final=True,
    )
    return b.build()


def gen_book_with_notes(rng: random.Random, idx: int) -> dict:
    """User volunteers a note → book_appointment with `notes` arg."""
    service = rng.choice(["Colore", "Pulizia Viso", "Trattamento Anti-età"])
    pro = rng.choice(SERVICE_TO_PROS[service])
    start, end = _date_range(rng, span_days=3)
    chosen_date = start
    chosen_time = _time_slot(rng)
    note = rng.choice([
        "Vorrei provare una nuance sul castano scuro.",
        "Ho la pelle sensibile, attenzione ai prodotti.",
        "Sono incinta, evitare prodotti aggressivi.",
        "È la mia prima volta da voi.",
    ])
    b = TraceBuilder(trace_id=f"realistic-book-notes-{idx:02d}")
    b.system()
    b.user(
        f"Posso prenotare un {service.lower()} con {pro}? "
        f"Volevo aggiungere una nota: {note}"
    )
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": start,
        "end_date_str": end,
        "professional": pro,
    })])
    b.tool_result(_availability_payload(service, start, end, rng, professional=pro))
    b.ai(
        f"Ho disponibilità il {chosen_date} alle {chosen_time} con {pro}. "
        f"Procedo con la prenotazione e aggiungo la nota?"
    )
    b.user("Sì grazie.")
    b.tool_calls([("book_appointment", {
        "service": service,
        "date_str": chosen_date,
        "time_str": chosen_time,
        "professional": pro,
        "notes": note,
    })])
    b.tool_result(_booking_confirmation(service, chosen_date, chosen_time, pro, rng))
    b.ai(
        f"Prenotazione confermata: {service} con {pro} il {chosen_date} alle "
        f"{chosen_time}. La nota è stata aggiunta. A presto!",
        is_final=True,
    )
    return b.build()


def gen_modify_appointment(rng: random.Random, idx: int) -> dict:
    """Customer wants to move an existing booking to a new day/time."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    old_date = _date_str(rng)
    old_time = _time_slot(rng)
    new_date = _date_str(rng)
    new_time = _time_slot(rng)
    b = TraceBuilder(trace_id=f"realistic-modify-{idx:02d}")
    b.system()
    b.user(
        f"Avevo prenotato {service.lower()} con {pro} il {old_date} alle {old_time}, "
        f"posso spostarlo al {new_date} alle {new_time}?"
    )
    b.tool_calls([("get_professional_availability", {
        "start_date_str": new_date,
        "end_date_str":   new_date,
        "professional":   pro,
    })])
    b.tool_result({"blocks": [{
        "date": new_date, "professional": pro,
        "free": [f"{new_time}-{int(new_time[:2])+1:02d}:{new_time[3:]} (1h)"],
    }]})
    b.tool_calls([("modify_appointment", {
        "original_service":      service,
        "original_date":         old_date,
        "original_time":         old_time,
        "original_professional": pro,
        "new_service":           service,
        "new_date":              new_date,
        "new_time":              new_time,
        "new_professional":      pro,
    })])
    b.tool_result({
        "status": "modified",
        "old": {"date": old_date, "time": old_time},
        "new": {"date": new_date, "time": new_time},
    })
    b.ai(
        f"Fatto! Ho spostato il tuo {service.lower()} con {pro} dal {old_date} "
        f"alle {old_time} al {new_date} alle {new_time}.",
        is_final=True,
    )
    return b.build()


def gen_cancel_then_rebook(rng: random.Random, idx: int) -> dict:
    """Customer cancels, then re-books a different service/day in same chat."""
    old_service = rng.choice(SERVICES)
    old_pro = rng.choice(SERVICE_TO_PROS[old_service])
    old_date = _date_str(rng)
    old_time = _time_slot(rng)
    new_service = rng.choice([s for s in SERVICES if s != old_service])
    new_pro = rng.choice(SERVICE_TO_PROS[new_service])
    new_date = _date_str(rng)
    new_time = _time_slot(rng)

    b = TraceBuilder(trace_id=f"realistic-cancel-rebook-{idx:02d}")
    b.system()
    b.user(
        f"Devo cancellare {old_service.lower()} con {old_pro} del {old_date} alle "
        f"{old_time} e prenotare invece un {new_service.lower()}."
    )
    b.tool_calls([("cancel_appointment", {
        "service": old_service,
        "date_str": old_date,
        "time_str": old_time,
        "professional": old_pro,
    })])
    b.tool_result({"status": "cancelled", "service": old_service, "date": old_date, "time": old_time})
    b.ai(
        f"Ho cancellato {old_service.lower()} con {old_pro} del {old_date}. "
        f"Per il {new_service.lower()} preferisci una data in particolare?"
    )
    b.user(f"Diciamo il {new_date} verso le {new_time}.")
    b.tool_calls([("get_service_availability", {
        "service": new_service,
        "start_date_str": new_date,
        "end_date_str":   new_date,
    })])
    b.tool_result(_availability_payload(new_service, new_date, new_date, rng))
    b.tool_calls([("book_appointment", {
        "service":      new_service,
        "date_str":     new_date,
        "time_str":     new_time,
        "professional": new_pro,
    })])
    b.tool_result(_booking_confirmation(new_service, new_date, new_time, new_pro, rng))
    b.ai(
        f"Tutto a posto: cancellazione fatta e nuova prenotazione per "
        f"{new_service.lower()} con {new_pro} il {new_date} alle {new_time}. "
        f"A presto!",
        is_final=True,
    )
    return b.build()


def gen_multi_person_booking(rng: random.Random, idx: int) -> dict:
    """User books two appointments at same time for two people → uses
    allow_concurrent_booking flag for the second call."""
    service_a = rng.choice(["Manicure", "Pedicure", "Pulizia Viso"])
    service_b = rng.choice([s for s in ("Manicure", "Pedicure", "Pulizia Viso") if s != service_a])
    pro_a = rng.choice(SERVICE_TO_PROS[service_a])
    pro_b = rng.choice([p for p in SERVICE_TO_PROS[service_b] if p != pro_a]) \
        if len(SERVICE_TO_PROS[service_b]) > 1 else SERVICE_TO_PROS[service_b][0]
    date_str = _date_str(rng)
    time_str = _time_slot(rng)

    b = TraceBuilder(trace_id=f"realistic-multi-person-{idx:02d}")
    b.system()
    b.user(
        f"Vorrei prenotare {service_a.lower()} per me e {service_b.lower()} per "
        f"mia sorella, in contemporanea il {date_str} alle {time_str}."
    )
    b.tool_calls([("get_service_availability", {
        "service": service_a,
        "start_date_str": date_str,
        "end_date_str":   date_str,
    })])
    b.tool_result(_availability_payload(service_a, date_str, date_str, rng))
    b.tool_calls([("book_appointment", {
        "service":      service_a,
        "date_str":     date_str,
        "time_str":     time_str,
        "professional": pro_a,
    })])
    b.tool_result(_booking_confirmation(service_a, date_str, time_str, pro_a, rng))
    b.tool_calls([("book_appointment", {
        "service":      service_b,
        "date_str":     date_str,
        "time_str":     time_str,
        "professional": pro_b,
        "allow_concurrent_booking": True,
    })])
    b.tool_result(_booking_confirmation(service_b, date_str, time_str, pro_b, rng))
    b.ai(
        f"Fatto! Ho prenotato {service_a.lower()} con {pro_a} per te e "
        f"{service_b.lower()} con {pro_b} per tua sorella, entrambi il "
        f"{date_str} alle {time_str}. Vi aspettiamo!",
        is_final=True,
    )
    return b.build()


def gen_info_then_book(rng: random.Random, idx: int) -> dict:
    """Customer asks for shop info first (hours, address), then proceeds
    to book — exercises mixing get_shop_info with the booking flow."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    date_str = _date_str(rng)
    time_str = _time_slot(rng)

    b = TraceBuilder(trace_id=f"realistic-info-then-book-{idx:02d}")
    b.system()
    b.user("Buongiorno! Volevo sapere dove siete e che orari fate, e poi vorrei prenotare.")
    b.tool_calls([("get_shop_info", {})])
    b.tool_result({
        "address": "Via Torino 42, Milano",
        "hours": {"lun-ven": "09:00-19:00", "sab": "09:00-17:00"},
    })
    b.ai(
        "Siamo in Via Torino 42 a Milano, aperti lun-ven 09:00-19:00 e sabato "
        "09:00-17:00. Per quale servizio vuoi prenotare?"
    )
    b.user(f"Un {service.lower()} con {pro} il {date_str} alle {time_str}.")
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": date_str,
        "end_date_str": date_str,
        "professional": pro,
    })])
    b.tool_result(_availability_payload(service, date_str, date_str, rng, professional=pro))
    b.tool_calls([("book_appointment", {
        "service":      service,
        "date_str":     date_str,
        "time_str":     time_str,
        "professional": pro,
    })])
    b.tool_result(_booking_confirmation(service, date_str, time_str, pro, rng))
    b.ai(
        f"Prenotazione confermata: {service.lower()} con {pro} il {date_str} "
        f"alle {time_str}. Ti aspettiamo in Via Torino 42!",
        is_final=True,
    )
    return b.build()


# ----- clarification scenarios -----

def gen_clarify_service(rng: random.Random, idx: int) -> dict:
    """User says 'voglio un trattamento' / 'taglio' (ambiguous) → AI asks
    for clarification BEFORE calling any tool."""
    ambiguous_q = rng.choice([
        "Vorrei prenotare un trattamento per il viso.",
        "Posso prenotare un taglio per la prossima settimana?",
        "Volevo qualcosa di rilassante per sabato.",
    ])
    if "viso" in ambiguous_q:
        chosen = rng.choice(["Pulizia Viso", "Trattamento Anti-età"])
    elif "taglio" in ambiguous_q:
        chosen = rng.choice(["Taglio Donna", "Taglio Uomo"])
    else:
        chosen = "Massaggio Rilassante"
    pro = rng.choice(SERVICE_TO_PROS[chosen])
    start, end = _date_range(rng, span_days=4)

    b = TraceBuilder(trace_id=f"realistic-clarify-service-{idx:02d}")
    b.system()
    b.user(ambiguous_q)
    b.ai(
        f"Per servirti al meglio, intendi un {chosen}? "
        f"Confermami così controllo subito la disponibilità."
    )
    b.user(f"Sì, esatto, {chosen.lower()}.")
    b.tool_calls([("get_service_availability", {
        "service": chosen,
        "start_date_str": start,
        "end_date_str": end,
    })])
    b.tool_result(_availability_payload(chosen, start, end, rng))
    b.ai(
        f"Per {chosen} ho disponibilità con {pro} tra il {start} e il {end}. "
        f"Vuoi che ti prenoti uno di questi orari?",
        is_final=True,
    )
    return b.build()


def gen_clarify_date(rng: random.Random, idx: int) -> dict:
    """User says 'la prossima settimana' / 'a fine mese' → AI asks for
    a precise date (DD-MM-YYYY) before calling availability."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    vague = rng.choice([
        "la prossima settimana",
        "a fine mese",
        "tra qualche giorno",
    ])
    start, end = _date_range(rng, span_days=3)
    b = TraceBuilder(trace_id=f"realistic-clarify-date-{idx:02d}")
    b.system()
    b.user(f"Vorrei un {service.lower()} con {pro} {vague}.")
    b.ai(
        "Certo! Mi puoi indicare un range di date più preciso (formato "
        "GG-MM-YYYY) così controllo la disponibilità di "
        f"{pro}?"
    )
    b.user(f"Diciamo dal {start} al {end}.")
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": start,
        "end_date_str": end,
        "professional": pro,
    })])
    b.tool_result(_availability_payload(service, start, end, rng, professional=pro))
    b.ai(
        f"Tra il {start} e il {end} {pro} ha disponibilità per {service.lower()}: "
        f"vuoi che ti prenoti il primo slot disponibile?",
        is_final=True,
    )
    return b.build()


def gen_clarify_professional(rng: random.Random, idx: int) -> dict:
    """User asks for service without professional → AI lists available pros."""
    service = rng.choice([s for s in SERVICES if len(SERVICE_TO_PROS[s]) > 1])
    pros = SERVICE_TO_PROS[service]
    chosen_pro = rng.choice(pros)
    start, end = _date_range(rng, span_days=3)
    b = TraceBuilder(trace_id=f"realistic-clarify-pro-{idx:02d}")
    b.system()
    b.user(f"Vorrei prenotare un {service.lower()} per la prossima settimana.")
    b.ai(
        f"Per {service} abbiamo {', '.join(pros[:-1])} e {pros[-1]}. "
        f"Hai una preferenza, o ti va bene il primo disponibile?"
    )
    b.user(f"Va bene {chosen_pro}.")
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": start,
        "end_date_str": end,
        "professional": chosen_pro,
    })])
    b.tool_result(_availability_payload(service, start, end, rng, professional=chosen_pro))
    b.ai(
        f"Perfetto, con {chosen_pro} ho disponibilità tra il {start} e il {end}. "
        f"Quale orario preferisci?",
        is_final=True,
    )
    return b.build()


# ----- edge-case scenarios -----

def gen_edge_slot_taken(rng: random.Random, idx: int) -> dict:
    """Customer asks for a specific slot, the booking tool reports it as
    taken, agent proposes alternatives via get_service_availability."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    date_str = _date_str(rng)
    requested_time = _time_slot(rng)
    alt_time = _time_slot(rng)
    while alt_time == requested_time:
        alt_time = _time_slot(rng)

    b = TraceBuilder(trace_id=f"realistic-edge-slot-taken-{idx:02d}")
    b.system()
    b.user(f"Mi prenoti un {service.lower()} con {pro} il {date_str} alle {requested_time}?")
    b.tool_calls([("book_appointment", {
        "service":      service,
        "date_str":     date_str,
        "time_str":     requested_time,
        "professional": pro,
    })])
    b.tool_result({
        "status": "error",
        "code":   "slot_unavailable",
        "message": f"{pro} non è disponibile il {date_str} alle {requested_time}.",
    })
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": date_str,
        "end_date_str":   date_str,
        "professional":   pro,
    })])
    b.tool_result(_availability_payload(service, date_str, date_str, rng, professional=pro))
    b.ai(
        f"Mi dispiace, {pro} il {date_str} alle {requested_time} è già occupato. "
        f"Però ho un altro slot alle {alt_time} dello stesso giorno: "
        f"vuoi che lo prenoti?",
        is_final=True,
    )
    return b.build()


def gen_edge_booking_not_found(rng: random.Random, idx: int) -> dict:
    """Customer asks to cancel a booking that doesn't exist."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    date_str = _date_str(rng)
    time_str = _time_slot(rng)
    b = TraceBuilder(trace_id=f"realistic-edge-not-found-{idx:02d}")
    b.system()
    b.user(
        f"Cancella la prenotazione di {service.lower()} con {pro} del "
        f"{date_str} alle {time_str}."
    )
    b.tool_calls([("cancel_appointment", {
        "service":      service,
        "date_str":     date_str,
        "time_str":     time_str,
        "professional": pro,
    })])
    b.tool_result({
        "status": "error",
        "code":   "booking_not_found",
        "message": "Nessuna prenotazione trovata con quei dati.",
    })
    b.ai(
        f"Non risulta nessuna prenotazione di {service.lower()} con {pro} il "
        f"{date_str} alle {time_str}. Controlla i dettagli e fammi sapere — "
        f"magari era con un altro professionista o in un altro orario?",
        is_final=True,
    )
    return b.build()


def gen_edge_tool_error(rng: random.Random, idx: int) -> dict:
    """An availability call returns a transient error → agent retries."""
    service = rng.choice(SERVICES)
    pro = rng.choice(SERVICE_TO_PROS[service])
    start, end = _date_range(rng, span_days=3)
    b = TraceBuilder(trace_id=f"realistic-edge-tool-error-{idx:02d}")
    b.system()
    b.user(f"Disponibilità per {service.lower()} con {pro} la prossima settimana?")
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": start,
        "end_date_str": end,
        "professional": pro,
    })])
    b.tool_result({
        "status": "error",
        "code": "service_unavailable",
        "message": "Servizio temporaneamente non raggiungibile.",
    })
    b.tool_calls([("get_service_availability", {
        "service": service,
        "start_date_str": start,
        "end_date_str": end,
        "professional": pro,
    })])
    b.tool_result(_availability_payload(service, start, end, rng, professional=pro))
    b.ai(
        f"Ecco le disponibilità per {service.lower()} con {pro} tra il {start} "
        f"e il {end}: vuoi che ti prenoti uno degli slot?",
        is_final=True,
    )
    return b.build()


# ── Scenario plan & main ──────────────────────────────────────────────────


# (generator, count) — totals to 50 with the targeted complexity mix.
SCENARIO_PLAN: tuple[tuple[callable, int], ...] = (
    # single-tool: 15 (~30%)
    (gen_single_shop_info,            5),
    (gen_single_service_availability, 4),
    (gen_single_pro_availability,     3),
    (gen_single_cancel,               3),
    # multi-tool: 24 (~48%)
    (gen_check_then_book,             8),
    (gen_book_with_notes,             3),
    (gen_modify_appointment,          5),
    (gen_cancel_then_rebook,          3),
    (gen_multi_person_booking,        2),
    (gen_info_then_book,              3),
    # clarification: 8 (~16%)
    (gen_clarify_service,             3),
    (gen_clarify_date,                3),
    (gen_clarify_professional,        2),
    # edge cases: 3 (~6%)
    (gen_edge_slot_taken,             1),
    (gen_edge_booking_not_found,      1),
    (gen_edge_tool_error,             1),
)


def main() -> None:
    rng = random.Random(SEED)
    records: list[dict] = []

    total = sum(n for _, n in SCENARIO_PLAN)
    if total != TARGET_TRACES:
        raise SystemExit(
            f"SCENARIO_PLAN sums to {total}, expected {TARGET_TRACES}. "
            f"Adjust the per-generator counts."
        )

    for gen, count in SCENARIO_PLAN:
        for i in range(count):
            records.append(gen(rng, i))

    rng.shuffle(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Coverage summary
    tool_counts: dict[str, int] = {}
    for rec in records:
        for msg in rec["full_trace"]:
            for tc in msg.get("tool_calls", []) or []:
                tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1

    print(f"Wrote {len(records)} records → {OUTPUT_PATH}")
    print("Tool-call coverage:")
    for name in sorted(tool_counts):
        print(f"  {name:32s} {tool_counts[name]:4d}")


if __name__ == "__main__":
    main()
