"""
Code for calling the language model to get choice logits
"""

from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from src.utils import get_logprobs_from_outputs, get_messages, preprocess_messages

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose a shape from a set of shapes. Your goal is to guess which of the shapes the describer is trying to get the matcher to choose. The shapes, with their labels, are shown in the image.
Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose, and no other text. You will receive feedback telling you whether your choice was correct or incorrect.
"""

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def get_logits(
    df: pd.DataFrame,
    model_name: str,
    llm: LLM,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
) -> list[pd.DataFrame]:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Collect all messages from all dataframes first
    all_messages = []

    sampling_params = SamplingParams(max_tokens=1, logprobs=1000, temperature=1)

    if n_trials is not None:
        df = df.head(n_trials)

    df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

    for chat_prompt in df["chat_prompt"]:
        messages = get_messages(
            SYSTEM_PROMPT, chat_prompt, include_image, grid_image, model_name
        )

        all_messages.append(messages)

    # Apply chat template to all messages
    print("Applying chat templates...")
    all_prompts = []
    for messages in tqdm(all_messages):
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if include_image:
            all_prompts.append(
                {"prompt": text, "multi_modal_data": {"image": grid_image}}
            )
        else:
            all_prompts.append({"prompt": text})

    print("Doing inference...")
    outputs = llm.generate(
        all_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    print("finished inference, getting logprobs...")

    all_choice_logprobs = get_logprobs_from_outputs(outputs, CHOICES)

    df["model_logprobs"] = all_choice_logprobs

    return df.drop(columns=["chat_prompt"])
