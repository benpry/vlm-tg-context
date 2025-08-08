import os
from argparse import ArgumentParser
from glob import glob

import pandas as pd
from PIL import Image
from pyprojroot import here

# from src.lm import get_logits
from src.lm_vllm import get_logits

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="the name of the model to evaluate",
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
        help="the number of trials to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="the batch size for processing (default: 32)",
    )

    args = parser.parse_args()

    data_filepaths = glob(str(here("context_prep/*.csv")))
    print("data filepaths:", data_filepaths)

    dfs = []
    for filepath in data_filepaths:
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    for filepath, df in zip(data_filepaths, dfs):
        output_path = filepath.replace(
            ".csv", f"_{args.model_name.split('/')[-1]}_logprobs.csv"
        )
        if os.path.exists(output_path):
            print(f"Skipping {filepath} as output file already exists.")
            continue

        print(f"Processing {filepath}...")
        df_results = get_logits(
            df,
            args.model_name,
            grid_image,
            n_trials=args.n_trials,
            batch_size=args.batch_size,
        )

        df.to_csv(here(output_path), index=False)
