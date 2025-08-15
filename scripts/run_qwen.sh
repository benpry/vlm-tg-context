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

cd ~/vlm-tg-context

MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"

python scripts/call_lm.py \
    --model $MODEL_NAME \
    --no_image
