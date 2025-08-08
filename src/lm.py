"""
Code for calling the language model to get choice logits
"""

from typing import Optional

import nest_asyncio
import pandas as pd
import sglang as sgl
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from src.utils import (
    get_logprobs_from_outputs_vllm,
    preprocess_messages,
)

nest_asyncio.apply()

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose an image from a list of images. Your goal is to guess which of the images the describer is trying to get the matcher to choose. The images, with their labels, are shown in the image.
ForCausalLM
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

    choice_token_ids = [
        processor.tokenizer.encode(choice, add_special_tokens=False)[0]
        for choice in CHOICES
    ]

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="kernels-community/flash-attn", trust_remote_code=True
    # )

    # Collect all messages from all dataframes first
    all_messages = []
    row_indices = []  # Track which row within each df

    llm = None  # load the language model lazily
    sampling_params = {"max_new_tokens": 1}

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
        row_indices.append(row_idx)

    # Apply chat template to all messages
    print("Applying chat templates...")
    all_texts = []
    for messages in tqdm(all_messages):
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        all_texts.append(text)

    print("Doing inference...")
    if llm is None:
        llm = sgl.Engine(
            model_path=model_name,
            device="cuda",
            dtype="bfloat16",
            cuda_graph_max_bs=8,
            context_length=4096,
            pp_size=1,
            tp_size=1,
            trust_remote_code=True,
        )

        # Test if it's working
        print("Model loaded successfully!")
        print(f"Model: {llm}")

        # Try a simple generation
        response = llm.generate("Hello, world!", sampling_params={"max_new_tokens": 10})
        print(f"Response: {response}")

    outputs = llm.generate(
        all_texts,
        sampling_params,
        image_data=[grid_image] * len(all_texts),
        return_logprob=True,
        top_logprobs_num=1000,
    )

    all_choice_logprobs = get_logprobs_from_outputs_vllm(
        outputs, CHOICES, choice_token_ids
    )

    df["model_logprobs"] = all_choice_logprobs

    return df[
        [
            "trial_id",
            "stage_num",
            "rep_num",
            "trial_num",
            "chat_prompt",
            "model_logprobs",
            "target",
        ]
    ]
