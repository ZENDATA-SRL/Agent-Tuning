import json


def train_test_split(dataset, test_size, eval_size, random_state):
    train_dataset = []
    test_dataset = []
    eval_dataset = []

    train_dataset = dataset[:int(len(dataset) * (1 - test_size - eval_size))]
    test_dataset = dataset[int(len(dataset) * (1 - test_size - eval_size)):int(len(dataset) * (1 - eval_size))]
    eval_dataset = dataset[int(len(dataset) * (1 - eval_size)):]
    return train_dataset, test_dataset, eval_dataset


dataset_name = "umore_july_dataset"
with open(f"data/{dataset_name}.json", "r") as f:
    dataset = json.load(f)


traces_w_tool_call = []
traces_wo_tool_call = []
traces_multi_tool_call = []

for trace in dataset:
    messages = trace["messages"]

    contains_tool_call = False
    multi_tool_call = 0
    for message in messages:
        if message.get("tool_calls", None) is not None:
            contains_tool_call = True
            multi_tool_call += 1
            if multi_tool_call > 1:
                break

    if contains_tool_call:
        traces_w_tool_call.append(trace)
        if multi_tool_call > 1:
            traces_multi_tool_call.append(trace)
    else:
        traces_wo_tool_call.append(trace)


train_dataset = []
test_dataset = []
eval_dataset = []

w_tool_call_train, w_tool_call_test, w_tool_call_eval = train_test_split(traces_w_tool_call, test_size=0.1, eval_size=0.1, random_state=42)
wo_tool_call_train, wo_tool_call_test, wo_tool_call_eval = train_test_split(traces_wo_tool_call, test_size=0.1, eval_size=0.1, random_state=42)
multi_tool_call_train, multi_tool_call_test, multi_tool_call_eval = train_test_split(traces_multi_tool_call, test_size=0.1, eval_size=0.1, random_state=42)

train_dataset = w_tool_call_train #+ wo_tool_call_train + multi_tool_call_test
test_dataset = w_tool_call_test #+ wo_tool_call_test + multi_tool_call_test
eval_dataset = w_tool_call_eval #+ wo_tool_call_eval + multi_tool_call_eval



# break each trace in multiple parts depends on the assistant response
def break_trace(trace):
    traces = []

    for i, m in enumerate(trace["messages"]):
        if m.get("role", None) == "assistant":
            # take every message before the assistant response and the assistant response
            part = trace["messages"][:i+1]
            traces.append({
                "tools": trace["tools"],
                "messages": part,
            })

    return traces

def break_dataset(dataset):
    traces = []
    for trace in dataset:
        traces.extend(break_trace(trace))
    return traces


train_dataset = break_dataset(train_dataset)
test_dataset = break_dataset(test_dataset)
eval_dataset = break_dataset(eval_dataset)


# create directory if not exists
import os
if not os.path.exists(f"data/{dataset_name}"):
    os.makedirs(f"data/{dataset_name}")

# save as jsonl
with open(f"data/{dataset_name}/train_only_tool_call.jsonl", "w") as f:
    for trace in train_dataset:
        json.dump(trace, f)
        f.write("\n")
with open(f"data/{dataset_name}/test_only_tool_call.jsonl", "w") as f:
    for trace in test_dataset:
        json.dump(trace, f)
        f.write("\n")
with open(f"data/{dataset_name}/eval_only_tool_call.jsonl", "w") as f:
    for trace in eval_dataset:
        json.dump(trace, f)
        f.write("\n")