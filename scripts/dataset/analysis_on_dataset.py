import json


with open("data/umore_july_dataset.json", "r") as f:
    dataset = json.load(f)

w_tool_call = 0
wo_tool_call = 0
w_multi_tool_call = 0


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
        w_tool_call += 1
        if multi_tool_call > 1:
            w_multi_tool_call += 1
    else:
        wo_tool_call += 1

print(f"w_tool_call: {w_tool_call}")
print(f"wo_tool_call: {wo_tool_call}")
print(f"w_multi_tool_call: {w_multi_tool_call}")