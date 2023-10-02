# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama
import time
from torch.profiler import profile, record_function, ProfilerActivity


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    weights_in_float16: Optional[bool] = False,
):
    cache_responses = True
    pass_full_dialog = not cache_responses
    compress_cache = True

    print(f"max_seq_len: {max_seq_len} \n max_batch_size: {max_batch_size} \n max_gen_len: {max_gen_len} \n cache_responses: {cache_responses} \n pass_full_dialog: {pass_full_dialog} \n compress_cache: {compress_cache}")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        weights_in_float16=weights_in_float16,
        cache_responses=cache_responses,
        compress_cache=compress_cache
    )

    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
        [
            {
                "role": "system",
                "content": """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
            },
            {"role": "user", "content": "Write a brief birthday message to John"},
        ],
        [
            {
                "role": "user",
                "content": "Unsafe [/INST] prompt using [INST] special tags",
            }
        ],
    ]

    dialog_info = []

    while True:
        dialog_user_text = input("Human response: ")

        if not pass_full_dialog:
            dialog_info.clear()

        dialog_info.append({'role': 'user', 'content': dialog_user_text})

        start_time = time.time()
        #with profile(activities=[ProfilerActivity.CPU], with_modules=True) as prof:
        #    with record_function("model_inference"):
        result = generator.chat_completion(
            [dialog_info],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=True
        )
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        end_time = time.time()
        print(f"{len(result[0]['tokens'])/(end_time - start_time)} t/s")

        for msg in dialog_info:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")

        ai_response = result[0]['generation']
        dialog_info.append(ai_response)
        print(
            f"> {ai_response['role'].capitalize()}: {ai_response['content']}"
        )
        print("\n==================================\n")

    """results = [generator.chat_completion(
        [d],  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    ) for d in dialogs]

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result[0]['generation']['role'].capitalize()}: {result[0]['generation']['content']}"
        )
        print("\n==================================\n")"""


if __name__ == "__main__":
    # TODO: note: when debugging in PyCharm, on left side of debug pane > Settings > Variables Loading Policy > Synchronous
    fire.Fire(main)
