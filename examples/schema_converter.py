import json

input_file = "/data01/qian_dev/datasets/sharegpt_train.jsonl"
output_file = "/data01/qian_dev/datasets/sharegpt_train_converted_all.jsonl"

MAX_ENTRIES = None   # ← 改这里，例如 100；None 表示全部

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system",
}

DEFAULT_SYSTEM = "You are a helpful assistant."

def convert_file(input_path, output_path):
    written = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if MAX_ENTRIES is not None and written >= MAX_ENTRIES:
                break

            if not line.strip():
                continue

            obj = json.loads(line)

            msgs = obj.get("conversations", [])
            new_msgs = []

            for m in msgs:
                role_raw = str(m.get("user", "")).lower()
                text = (m.get("text") or "").strip()
                if not text:
                    continue

                role = ROLE_MAP.get(role_raw, "user")
                new_msgs.append({"role": role, "content": text})

            if not new_msgs:
                continue

            if new_msgs[0]["role"] != "system":
                new_msgs.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})

            fout.write(json.dumps({"conversations": new_msgs}, ensure_ascii=False) + "\n")
            written += 1

    print("written:", written)


if __name__ == "__main__":
    convert_file(input_file, output_file)