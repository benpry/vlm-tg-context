"""
Quick test script for the SGLang implementation
"""
# launch the offline engine
import asyncio
import io
import os

from PIL import Image
import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()


if __name__ == "__main__":
    llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1}

    outputs = llm.generate(prompts, sampling_params, return_logprob=True, top_logprobs_num=100)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        print(output)