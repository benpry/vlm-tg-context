"""
Code for the "interactive" version of the language model evaluation, where the model gets limited
feedback on its own choices rather than human responses.
"""

from ast import literal_eval
from typing import Optional

import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from src.lm import CHOICES, SYSTEM_PROMPT
from src.utils import get_logprobs_from_outputs, get_messages, preprocess_messages


def get_logprobs_and_predictions(prompts, sampling_params, llm):

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    all_choice_logprobs = get_logprobs_from_outputs(outputs, CHOICES)

    predictions = []
    for choice_logprobs in all_choice_logprobs:
        predictions.append(max(choice_logprobs, key=choice_logprobs.get))

    return all_choice_logprobs, predictions


def update_histories(df: pd.DataFrame, trial_num: int):
    if trial_num == df["trialNum"].max():
        return

    df_nextround = df[df["trialNum"] == trial_num + 1][
        ["gameId", "selection_history", "correctness_history", "target_history"]
    ]
    df_thisround = df[df["trialNum"] == trial_num][["gameId", "model_prediction"]]

    df_nextround = df_nextround.merge(df_thisround, on="gameId", how="left")

    # update the selection history
    df_nextround["selection_history"] = df_nextround.apply(
        lambda x: x["selection_history"] + [x["model_prediction"]],
        axis=1,
    )
    # update the correctness history
    df_nextround["correctness_history"] = df_nextround.apply(
        lambda x: x["correctness_history"]
        + [x["model_prediction"] == x["target_history"][-1]],
        axis=1,
    )

    # update the dataframe's selection and correctness histories
    df.loc[df_nextround.index, "selection_history"] = df_nextround["selection_history"]
    df.loc[df_nextround.index, "correctness_history"] = df_nextround[
        "correctness_history"
    ]


def prepare_round_prompts(
    df: pd.DataFrame,
    processor,
    include_image: bool,
    grid_image: Image.Image,
    model_name: str,
):
    all_prompts = []
    df["chat_prompt"] = df.apply(preprocess_messages, axis=1)
    for chat_prompt in df["chat_prompt"]:
        messages = get_messages(
            SYSTEM_PROMPT, chat_prompt, include_image, grid_image, model_name
        )

        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        if include_image:
            all_prompts.append(
                {"prompt": text, "multi_modal_data": {"image": grid_image}}
            )
        else:
            all_prompts.append({"prompt": text})

    return all_prompts


def run_interactive_evaluation(
    df: pd.DataFrame,
    model_name: str,
    llm: LLM,
    grid_image: Image.Image,
    include_image: bool = True,
    n_trials: Optional[int] = None,
) -> list[pd.DataFrame]:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    sampling_params = SamplingParams(max_tokens=1, logprobs=1000, temperature=1)

    if n_trials is not None:
        df = df.head(n_trials)

    if "trialNum" not in df.columns:
        df["trialNum"] = df["matcher_trialNum"]

    df["selection_history"] = df["selection_history"].apply(
        lambda x: literal_eval(x.replace("null", '"no response"'))
    )
    df["correctness_history"] = df["correctness_history"].apply(
        lambda x: literal_eval(x.replace("true", "True").replace("false", "False"))
    )

    for trial_num in range(df["trialNum"].max()):
        df_round = df[df["trialNum"] == trial_num]

        prompts = prepare_round_prompts(
            df_round, processor, include_image, grid_image, model_name
        )

        choice_logprobs, predictions = get_logprobs_and_predictions(
            prompts, sampling_params, llm
        )

        # save the logprobs to the dataframe
        df.loc[df_round.index, "model_logprobs"] = choice_logprobs
        df.loc[df_round.index, "model_prediction"] = predictions

        # update the selection and correctness histories
        update_histories(df, trial_num)

    return df
    return df
