from argparse import ArgumentParser
import json
import os


def task_instruction_path(task_name):
    """Path to the existing task instruction JSON, preferring task_instruction/left/."""
    for sub in ("task_instruction/left", "task_instruction"):
        p = f"./{sub}/{task_name}.json"
        if os.path.exists(p):
            return p
    return f"./task_instruction/{task_name}.json"


def clear_seen_unseen(task_name):
    path = task_instruction_path(task_name)
    with open(path, "r") as f:
        task_info_json = f.read()
    # print(task_info_json)
    task_info = json.loads(task_info_json)
    task_info["seen"] = []
    task_info["unseen"] = []
    with open(path, "w") as f:
        json.dump(task_info, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task_name", type=str, default="beat_block_hammer")
    args = parser.parse_args()
    clear_seen_unseen(args.task_name)
