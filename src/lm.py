"""
Code for calling the language model to get choice logits
"""

from typing import Optional

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_image_token, preprocess_messages

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.

Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose.
"""

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def get_logits(
    dfs: list[pd.DataFrame],
    model_name: str,
    grid_image: Image.Image,
    n_trials: Optional[int] = None,
) -> list[pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    choice_token_ids = [
        tokenizer.encode(choice, add_special_tokens=False)[0] for choice in CHOICES
    ]

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    return_dfs = []
    for df in dfs:
        if n_trials is not None:
            df = df.head(n_trials)

        df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

        prompts = []
        for chat_prompt in df["chat_prompt"]:
            chat = (
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT + get_image_token(model_name),
                    },
                    *chat_prompt,
                ],
            )
            print(f"chat: {chat}")
            prompt = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )[0]
            prompts.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": grid_image},
                }
            )

        out = model(
            prompts,
            return_dict=True,
        )

        logits = out.logits
        choice_logits = [logits[:, -1, tid] for tid in choice_token_ids]

        df["model_logits"] = choice_logits

        return_dfs.append(
            df[
                [
                    "trial_id",
                    "stage_num",
                    "rep_num",
                    "trial_num",
                    "chat_prompt",
                    "model_logits",
                    "target",
                ]
            ]
        )

    return return_dfs
