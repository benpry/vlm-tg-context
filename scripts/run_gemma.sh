#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-output/vtc_%j.out
#SBATCH --error=slurm-output/vtc_%j.err


source ~/.zshrc

conda activate vtc

export LD_LIBRARY_PATH="/scr/benpry/conda/envs/vtc/lib64:/scr/benpry/conda/envs/vtc/lib:$LD_LIBRARY_PATH"

cd ~/vlm-tg-context

MODEL_NAME="google/gemma-3-27b-it"

python scripts/call_lm.py \
    --model $MODEL_NAME
