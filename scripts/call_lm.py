from argparse import ArgumentParser

import pandas as pd
from PIL import Image
from pyprojroot import here

from src.lm import get_logits

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_filepaths",
        type=str,
        default="data/trials_with_history.csv",
        help="semicolon-separated list of data filepaths",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
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

    dfs = []
    for filepath in args.data_filepaths.split(";"):
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    result_dfs = get_logits(
        dfs,
        args.model_name,
        grid_image,
        n_trials=args.n_trials,
        batch_size=args.batch_size,
    )
    
    # Save results
    for i, df in enumerate(result_dfs):
        output_path = f"data/logits_results_{i}.csv"
        df.to_csv(here(output_path), index=False)
        print(f"Results saved to {output_path}")
