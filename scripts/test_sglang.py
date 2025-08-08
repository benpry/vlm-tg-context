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
from sglang.lang.chat_template import get_chat_template
from pyprojroot import here

from transformers import AutoProcessor

if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()


if __name__ == "__main__":
    llm = sgl.Engine(model_path="Qwen/Qwen2.5-VL-7B-Instruct")

    chat_template = get_chat_template("qwen2-vl")
    image_token = chat_template.image_token
    grid_image = Image.open(here("data/compiled_grid.png"))

    print(f"image token: {image_token}")

    messages = [
        {
            "role": "system", 
            "content":  f"{image_token}Please describe this image."
        }
    ]
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1}

    print(f"prompt before template: {messages}")
    prompt = chat_template.get_prompt(messages)

    print(f"prompt: {prompt}")

    output = llm.generate([prompt, prompt], sampling_params, image_data=grid_image, return_logprob=True, top_logprobs_num=100)

    print(prompt)
    print(output["text"])
    print(output.keys())
    print(output["meta_info"])
    print(output["meta_info"]["output_top_logprobs"][0])

    # for prompt, output in zip(prompts, outputs):
    #     print("===============================")
    #     print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    #     print(output)