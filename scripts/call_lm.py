import os
from argparse import ArgumentParser
from glob import glob

import pandas as pd
from PIL import Image
from pyprojroot import here
from vllm import LLM

from src.interactive import run_interactive_evaluation
from src.lm import get_logits


def get_dtype(model_name: str, float32: bool) -> str:
    if "kimi" in model_name.lower():
        # kimi won't work in float32
        return "float16" if float32 else "bfloat16"
    else:
        return "float32" if float32 else "bfloat16"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="the name of the model to evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="human_history",
        help="the system prompt to use",
    )
    parser.add_argument(
        "--grid_image_path",
        type=str,
        default="data/compiled_grid.png",
        help="the path to the image of the compiled tangrams",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="the number of trials to evaluate on (default: all)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="the number of tensor parallel workers to use",
    )
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    data_filepaths = glob(str(here(f"context_prep/{args.data_dir}/*.csv")))
    print("data filepaths:", data_filepaths)

    dfs = []
    for filepath in data_filepaths:
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    llm = LLM(
        model=args.model_name,
        dtype=get_dtype(args.model_name, args.float32),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=7000,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 1},
        max_logprobs=1000,
        trust_remote_code=True,
        enforce_eager=args.float32,
        gpu_memory_utilization=0.95,
    )

    for filepath, df in zip(data_filepaths, dfs):
        output_path = filepath.replace(
            ".csv",
            f"_{args.model_name.split('/')[-1]}_logprobs{'_no_image' if args.no_image else ''}{'_float32' if args.float32 else ''}.csv",
        ).replace("context_prep", "data/logprobs")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {filepath} as output file already exists.")
            continue

        print(f"Processing {filepath}...")
        if args.interactive:
            df_results = run_interactive_evaluation(
                df,
                args.model_name,
                llm,
                grid_image,
                include_image=not args.no_image,
                n_trials=args.n_trials,
            )
        else:
            df_results = get_logits(
                df,
                args.model_name,
                llm,
                grid_image,
                include_image=not args.no_image,
                n_trials=args.n_trials,
            )

        df_results.to_csv(here(output_path), index=False)
