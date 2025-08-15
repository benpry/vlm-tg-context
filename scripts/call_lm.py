import os
from argparse import ArgumentParser
from glob import glob

from vllm import LLM
import pandas as pd
from PIL import Image
from pyprojroot import here

from src.lm import get_logits

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
    parser.add_argument("--no_image", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    data_filepaths = glob(str(here("context_prep/*.csv")))
    print("data filepaths:", data_filepaths)

    dfs = []
    for filepath in data_filepaths:
        dfs.append(pd.read_csv(here(filepath)))

    grid_image = Image.open(here(args.grid_image_path))

    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        tensor_parallel_size=2,
        max_model_len=8192,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 1},
        max_logprobs=1000,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
    )

    for filepath, df in zip(data_filepaths, dfs):
        output_path = filepath.replace(
            ".csv", f"_{args.model_name.split('/')[-1]}_logprobs_{'no_image' if args.no_image else 'with_image'}.csv"
        ).replace("context_prep", "data/logprobs")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {filepath} as output file already exists.")
            continue

        print(f"Processing {filepath}...")
        df_results = get_logits(
            df,
            args.model_name,
            llm,
            grid_image,
            include_image=not args.no_image,
            n_trials=args.n_trials,
            batch_size=args.batch_size,
        )

        df_results.to_csv(here(output_path), index=False)
