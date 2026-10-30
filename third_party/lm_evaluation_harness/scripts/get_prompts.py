import pdb

import argparse
from lm_eval import tasks


def get_prompts(tname):
    task = tasks.get_task_dict([tname])[tname]

    task.build_all_requests(
        limit=1,
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        tokenizer_name="",
    )

    for inst in task.instances:
        print(type(inst.arguments))
        print(inst.arguments[0])
        pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task name")
    args = parser.parse_args()

    get_prompts(args.task)
