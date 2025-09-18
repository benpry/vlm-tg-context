#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=slurm-output/vtc_%j.out
#SBATCH --error=slurm-output/vtc_%j.err
#SBATCH --constraint=[80G|141G]

source ~/.zshrc
cd ~/vlm-tg-context

# install conda if needed, create vtc environment if needed, activate vtc
source scripts/set_up_conda.sh

source ~/.zshrc
cd ~/vlm-tg-context
conda activate vtc

MODEL_NAME="google/gemma-3-27b-it"

python scripts/call_lm.py \
    --model_name $MODEL_NAME \
    --tensor_parallel_size 2 \
    --data_dir full_feedback \
    --interactive \
    --overwrite
