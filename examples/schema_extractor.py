import json

jsonl_file = "/data01/qian_dev/datasets/sharegpt_train_converted.jsonl"


def schema(x, depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        return

    if isinstance(x, dict):
        for k, v in x.items():
            print(f"{indent}{k}: {type(v).__name__}")
            schema(v, depth + 1, max_depth)

    elif isinstance(x, list):
        print(f"{indent}[list]")
        if x:
            schema(x[0], depth + 1, max_depth)


with open(jsonl_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 3:   # inspect first few samples
            break
        obj = json.loads(line)
        print("\n==== sample", i, "====")
        schema(obj)