"""
Code for calling the language model to get choice logits
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.utils import (
    convert_to_google_genai_style,
    get_logprobs_from_openai_choice,
    get_openai_messages,
    preprocess_messages,
)

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose a shape from a set of shapes. Your goal is to guess which of the shapes the describer is trying to get the matcher to choose. The shapes, with their labels, are shown in the image.
Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose, and no other text. You will receive feedback telling you whether your choice was correct or incorrect.
"""


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def get_completion_with_backoff(client, model, messages):
    if "gemini" in model.lower():
        # use the google genai client
        genai_messages, system_instruction = convert_to_google_genai_style(messages)
        return client.models.generate_content(
            model=model,
            contents=genai_messages,
            system_instruction=system_instruction,
            geneartion_config=types.GenerateContentConfig(
                response_logprobs=True,
                logprobs=20,
                max_output_tokens=1,
                temperature=1
            ),
        )
    else:
        # we're using an openai-style client
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            temperature=1,
            logprobs=True,
            top_logprobs=1000,
        )


def get_logits_single_row(
    client: OpenAI,
    model_name: str,
    messages: list,
) -> dict:
    response = get_completion_with_backoff(
        client=client,
        model=model_name,
        messages=messages,
    )
    return get_logprobs_from_openai_choice(response.choices[0], CHOICES)


REQUIRED_COLUMNS = [
    "message_history",
    "selection_history",
    "correctness_history",
    "message",
]


def get_logits(
    df: pd.DataFrame,
    model_name: str,
    client: OpenAI,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
) -> pd.DataFrame:
    # Validate required columns exist
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "For yoked/limited feedback evaluation, ensure your data includes "
            "message_history, selection_history, correctness_history, and message columns."
        )

    if n_trials is not None:
        df = df.head(n_trials)

    df["chat_prompt"] = df.apply(preprocess_messages, axis=1)

    print("Preparing messages...")
    all_messages = [
        get_openai_messages(SYSTEM_PROMPT, chat_prompt, include_image, grid_image)
        for chat_prompt in df["chat_prompt"]
    ]

    print("Doing inference...")

    with ThreadPoolExecutor(max_workers=20) as executor:
        all_choice_logprobs = list(
            tqdm(
                executor.map(
                    lambda msgs: get_logits_single_row(client, model_name, msgs),
                    all_messages,
                ),
                total=len(all_messages),
            )
        )

    df["model_logprobs"] = all_choice_logprobs

    return df.drop(columns=["chat_prompt"])
