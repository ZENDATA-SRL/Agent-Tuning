import json
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.api.core.request_options import RequestOptions
from tqdm import tqdm

load_dotenv()

# After download data from langfuse traces launch this script to add also system prompt to the dataset
langfuse = get_client()
traces = langfuse.api.trace.list(limit=1)
langfuse.api.trace.get(traces.data[0].id)
langfuse.api.observations.get_many(trace_id=traces.data[0].id)
export_name = "export-05-18"
output_data = []
with open(f'data/{export_name}.jsonl', 'r') as file:
    for line in tqdm(file):
        data = json.loads(line)
        if not data["output"]:
            continue
        
        trace_id = data['id']
        trace = langfuse.api.trace.get(trace_id=trace_id, request_options=RequestOptions(timeout_in_seconds=20, max_retries=10))
        system_prompt = None
        for observation in trace.observations:
            if observation.type == "GENERATION" and observation.input:
                for inp in observation.input:
                    if inp.get('role') == 'system':
                        system_prompt = inp.get('content')
                        break
            if system_prompt:
                messages = json.loads(data["output"])["messages"]
                messages = [{"role": message["type"], **message} for message in messages]
                output_data.append({
                    'trace_id': trace_id,
                    'full_trace': [{'role': 'system', 'content': system_prompt}, *messages],
                    'final_answer': messages[-1].get('content')
                })
                break
                
with open('data/traces-2026-05-18.jsonl', 'a') as file:
    for item in tqdm(output_data):
        file.write(json.dumps(item) + '\n')