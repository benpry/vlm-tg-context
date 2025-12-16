import os
from argparse import ArgumentParser
from glob import glob

import pandas as pd
from openai import OpenAI
from PIL import Image
from pyprojroot import here

from src.interactive import run_interactive_evaluation
from src.lm import get_logits

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="the name of the model to evaluate",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="the directory to read data from",
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
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--yoked",
        action="store_true",
        help="Run batch evaluation with histories yoked to human selections (limited feedback)",
    )
    parser.add_argument("--api_base", type=str, default=None, help="API Base URL")

    args = parser.parse_args()

    # Handle --yoked flag as alias for human_history data
    if args.yoked:
        if args.data_dir is not None:
            print("Warning: --yoked flag overrides --data_dir")
        args.data_dir = "human_history"

    data_filepaths = glob(str(here(f"context_prep/{args.data_dir}/*.csv")))
    print("data filepaths:", data_filepaths)

    dfs = []
    for filepath in data_filepaths:
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    client = OpenAI(
        base_url=args.api_base,
    )
    if "fireworks" in args.api_base:
        client.api_key = os.getenv("FIREWORKS_API_KEY")
    elif "google" in args.api_base:
        client.api_key = os.getenv("GOOGLE_API_KEY")
    elif "together" in args.api_base:
        client.api_key = os.getenv("TOGETHER_API_KEY")

    for filepath, df in zip(data_filepaths, dfs):
        output_path = filepath.replace(
            ".csv",
            f"_{args.model_name.split('/')[-1]}_logprobs{'_no_image' if args.no_image else ''}.csv",
        ).replace("context_prep", "data/logprobs")
        if args.interactive:
            output_path = output_path.replace(args.data_dir, "interactive")
        elif args.yoked:
            output_path = output_path.replace("human_history", "limited-feedback-yoked")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {filepath} as output file already exists.")
            continue

        print(f"Processing {filepath}...")
        if args.interactive:
            df_results = run_interactive_evaluation(
                df,
                args.model_name,
                client,
                grid_image,
                include_image=not args.no_image,
                n_trials=args.n_trials,
            )
        else:
            df_results = get_logits(
                df,
                args.model_name,
                client,
                grid_image,
                include_image=not args.no_image,
                n_trials=args.n_trials,
            )

        print(f"Saving {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_results.to_csv(here(output_path), index=False)
