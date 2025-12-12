"""
Code for calling the language model to get choice logits
"""

from typing import Optional

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import get_openai_messages, get_logprobs_from_openai_choice, preprocess_messages

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

SYSTEM_PROMPT = """You will be presented with a list of messages between people playing a reference game, where the describer has to get the matcher to choose a shape from a set of shapes. Your goal is to guess which of the shapes the describer is trying to get the matcher to choose. The shapes, with their labels, are shown in the image.
Please answer with just the letter corresponding to the image you think the describer is trying to get the matcher to choose, and no other text. You will receive feedback telling you whether your choice was correct or incorrect.
"""

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(10))
def get_completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_logits_single_row(
    client: OpenAI,
    model_name: str,
    messages: list,
) -> dict:
    response = get_completion_with_backoff(
        client=client,
        model=model_name,
        messages=messages,
        max_tokens=1,
        temperature=1,
        logprobs=True,
        top_logprobs=20,
    )
    return get_logprobs_from_openai_choice(response.choices[0], CHOICES)


def get_logits(
    df: pd.DataFrame,
    model_name: str,
    client: OpenAI,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
) -> pd.DataFrame:
    
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
