"""
Code for calling the language model to get choice logits
"""

from typing import Optional
import asyncio
import io
import os

import pandas as pd
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from kernels import get_kernel


import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

from src.utils import get_image_token, preprocess_messages


if is_in_ci():
    import patch
else:
    import nest_asyncio

    nest_asyncio.apply()

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
    batch_size: int = 8,
) -> list[pd.DataFrame]:
    processor = AutoProcessor.from_pretrained(model_name)

    choice_token_ids = [
        processor.tokenizer.encode(choice, add_special_tokens=False)[0] for choice in CHOICES
    ]

    flash_attn = get_kernel("kernels-community/flash-attn")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="kernels-community/flash-attn", trust_remote_code=True
    )

    # Collect all messages from all dataframes first
    all_messages = []
    df_indices = []  # Track which df each message belongs to
    row_indices = []  # Track which row within each df
    
    return_dfs = []
    for df_idx, df in enumerate(dfs):
        if n_trials is not None:
            df = df.head(n_trials)
        
        df["chat_prompt"] = df.apply(preprocess_messages, axis=1)
        
        for row_idx, chat_prompt in enumerate(df["chat_prompt"]):
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {"type": "image", "image": grid_image},
                    ],
                },
                *chat_prompt,
            ]
            all_messages.append(messages)
            df_indices.append(df_idx)
            row_indices.append(row_idx)

        # Process vision info once for all messages (they all use the same image)
        print("Processing vision information...")
        image_inputs, _ = process_vision_info(all_messages[0])

        # Apply chat template to all messages
        print("Applying chat templates...")
        all_texts = []
        for messages in tqdm(all_messages):
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            all_texts.append(text)

        # Process all messages in batches
        all_choice_logits = []
        print("Processing inference in batches...")
        
        for i in tqdm(range(0, len(all_messages), batch_size)):
            batch_texts = all_texts[i:i+batch_size]
            batch_size_actual = len(batch_texts)

            # Extract the corresponding image/video inputs for this batch
            batch_image_inputs = image_inputs * batch_size_actual

            # Process batch
            inputs = processor(
                text=batch_texts,
                images=batch_image_inputs,
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(model.device)

            with torch.inference_mode():
                out = model(**inputs)

            del inputs

            logits = out.logits
            
            # Extract choice logits for each item in the batch
            for j in range(batch_size_actual):
                choice_logits = [logits[j, -1, tid].item() for tid in choice_token_ids]
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
