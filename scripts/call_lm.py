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

    args = parser.parse_args()

    dfs = []
    for filepath in args.data_filepaths.split(";"):
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    get_logits(
        dfs,
        args.model_name,
        grid_image,
    )
