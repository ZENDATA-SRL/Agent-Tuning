import json
from collections import defaultdict

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv(".env")
langfuse = get_client()

filters = [
    {
        "type": "string",
        "column": "environment",
        "operator": "=",
        "value": "production",
    },
    {# start_time is after 2026-07-01
        "type": "datetime",
        "column": "startTime",
        "operator": ">=",
        "value": "2026-01-01",
    },
    {# end_time is before 2026-07-31
        "type": "datetime",
        "column": "endTime",
        "operator": "<=",
        "value": "2026-09-22",
    }
    # {
    #     "type": "string",
    #     "column": "sessionId",
    #     "operator": "=",
    #     "value": "9d5d998b-225c-4845-8728-2c5ffdbb26ea",
    # },
]

observations = []
cursor = None

while True:
    page = langfuse.api.observations.get_many(
        cursor=cursor,
        fields="core,basic,io",
        limit=1000,
        filter=json.dumps(filters),
    )

    observations.extend(
        {
            "id": observation.id,
            "trace_id": observation.trace_id,
            "start_time": observation.start_time,
            "end_time": observation.end_time,
            "type": observation.type,
            "name": observation.name,
            "input": observation.input,
            "output": observation.output,
        }
        for observation in page.data
    )

    cursor = page.meta.cursor
    if cursor is None:
        break


def decode_io(value):
    """Decode JSON-encoded Langfuse input/output when possible."""
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def get_field(observation, name):
    if isinstance(observation, dict):
        return observation.get(name)
    return getattr(observation, name)


def serialize_datetime(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


traces = defaultdict(list)
for observation in observations:
    trace_id = get_field(observation, "trace_id")
    if trace_id is None:
        continue

    start_time = get_field(observation, "start_time")
    end_time = get_field(observation, "end_time")
    traces[trace_id].append(
        {
            "id": get_field(observation, "id"),
            "parent_observation_id": get_field(observation, "parent_observation_id"),
            "type": get_field(observation, "type"),
            "name": get_field(observation, "name"),
            "start_time": serialize_datetime(start_time),
            "end_time": serialize_datetime(end_time),
            "input": decode_io(get_field(observation, "input")),
            "output": decode_io(get_field(observation, "output")),
        }
    )

for trace_observations in traces.values():
    trace_observations.sort(key=lambda observation: observation["start_time"])

complete_traces = [
    {
        "trace_id": trace_id,
        "observations": trace_observations,
    }
    for trace_id, trace_observations in traces.items()
]

with open("langfuse_dataset.json", "w") as f:
    f.write(json.dumps(complete_traces, ensure_ascii=False, indent=2))
