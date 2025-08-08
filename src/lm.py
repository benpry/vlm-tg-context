"""
Code for calling the language model to get choice logits
"""

from typing import Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from src.utils import (
    get_image_token,
    get_logprobs_from_outputs,
    preprocess_messages,
)

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.
Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose.
"""

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def get_logits(
    df: pd.DataFrame,
    model_name: str,
    grid_image: Image.Image,
    n_trials: Optional[int] = None,
    batch_size: int = 8,
) -> list[pd.DataFrame]:
    processor = AutoProcessor.from_pretrained(model_name)

    # Collect all messages from all dataframes first
    all_messages = []
    row_indices = []  # Track which row within each df

    llm = None  # load the language model lazily
    sampling_params = SamplingParams(max_tokens=1, logprobs=1000, temperature=1)

    if n_trials is not None:
        df = df.head(n_trials)

    df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

    for row_idx, chat_prompt in enumerate(df["chat_prompt"]):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + get_image_token(model_name),
            },
            *chat_prompt,
        ]
        all_messages.append(messages)
        row_indices.append(row_idx)

    # Apply chat template to all messages
    print("Applying chat templates...")
    all_prompts = []
    for messages in tqdm(all_messages):
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        all_prompts.append({"prompt": text, "multi_modal_data": {"image": grid_image}})

    print("Doing inference...")
    if llm is None:
        llm = LLM(
            model=model_name,
            dtype=torch.bfloat16,
            tensor_parallel_size=2,
            max_model_len=8192,
            max_num_seqs=8,
            max_logprobs=1000,
        )

    outputs = llm.generate(
        all_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    print("finished inference, getting logprobs...")

    all_choice_logprobs = get_logprobs_from_outputs(outputs, CHOICES)

    df["model_logprobs"] = all_choice_logprobs

    return df.drop(columns=["chat_prompt"])
