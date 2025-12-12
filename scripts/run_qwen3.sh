#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops,sc-loprio
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-output/qwen_%j.out
#SBATCH --error=slurm-output/qwen_%j.err

source ~/.zshrc
cd ~/vlm-tg-context

# install conda if needed, create vtc environment if needed, activate vtc
source scripts/set_up_uv.sh

MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"

python scripts/call_lm.py \
    --model_name $MODEL_NAME \
    --data_dir full_feedback \
    --api_base https://api.together.xyz/v1
