"""
Code for calling the language model to get choice logits
"""

from typing import Optional

import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from src.utils import get_image_token, preprocess_messages

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.
ForCausalLM
Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose.
"""

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def get_logits(
    dfs: list[pd.DataFrame],
    model_name: str,
    grid_image: Image.Image,
    n_trials: Optional[int] = None,
) -> list[pd.DataFrame]:
    processor = AutoProcessor.from_pretrained(model_name)

    choice_token_ids = [
        processor.tokenizer.encode(choice, add_special_tokens=False)[0] for choice in CHOICES
    ]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    return_dfs = []
    for df in dfs:
        if n_trials is not None:
            df = df.head(n_trials)

        df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

        all_inputs = []
        all_choice_logits = []
        print("Processing prompts...")
        for chat_prompt in tqdm(df["chat_prompt"]):
            messages = (
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                            {"type": "image", "image": grid_image},
                        ],
                    },
                    *chat_prompt,
                ],
            )

            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )[0]
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding = True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                out = model(
                    **inputs,
                )


            logits = out.logits
            choice_logits = [logits[0, -1, tid].item() for tid in choice_token_ids]
            all_choice_logits.append(choice_logits)

        df["model_logits"] = all_choice_logits

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
