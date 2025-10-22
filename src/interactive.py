"""
Code for the interactive version of the language model evaluation, where the model gets limited
feedback on its own choices rather than human responses.
"""

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

    # Get the original indices of future rounds before filtering
    future_rounds_mask = df["trialNum"] > trial_num
    future_rounds_indices = df[future_rounds_mask].index

    df_future_rounds = df.loc[future_rounds_mask][
        [
            "gameId",
            "trialNum",
            "selection_history",
            "correctness_history",
            "target_history",
        ]
    ].copy()
    df_thisround = df[df["trialNum"] == trial_num][["gameId", "model_prediction"]]

    df_future_rounds = df_future_rounds.merge(df_thisround, on="gameId", how="left")

    print(f"df future rounds length: {len(df_future_rounds)}")
    print(f"future indices length: {len(future_rounds_indices)}")

    # update the selection history
    df_future_rounds["selection_history"] = df_future_rounds.apply(
        lambda x: x["selection_history"] + [x["model_prediction"]],
        axis=1,
    )
    # update the correctness history
    df_future_rounds["correctness_history"] = df_future_rounds.apply(
        lambda x: x["correctness_history"]
        + [x["model_prediction"] == x["target_history"][x["trialNum"] - 1]],
        axis=1,
    )

    # update the dataframe's selection and correctness histories using original indices
    df.loc[future_rounds_indices, "selection_history"] = df_future_rounds[
        "selection_history"
    ].values
    df.loc[future_rounds_indices, "correctness_history"] = df_future_rounds[
        "correctness_history"
    ].values


def prepare_round_prompts(
    df_round: pd.DataFrame,
    processor,
    include_image: bool,
    grid_image: Image.Image,
    model_name: str,
):
    all_prompts = []
    df_round["chat_prompt"] = df_round.apply(preprocess_messages, axis=1)
    for chat_prompt in df_round["chat_prompt"]:
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

    df["selection_history"] = [[] for _ in range(len(df))]
    df["correctness_history"] = [[] for _ in range(len(df))]

    for trial_num in range(df["trialNum"].max() + 1):
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
