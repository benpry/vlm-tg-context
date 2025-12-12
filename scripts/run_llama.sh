#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops,sc-loprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=slurm-output/llama_%j.out
#SBATCH --error=slurm-output/llama_%j.err
#SBATCH --constraint=[80G|141G]

source ~/.zshrc
cd ~/vlm-tg-context

# install conda if needed, create vtc environment if needed, activate vtc
source scripts/set_up_uv.sh

MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"

python scripts/call_lm.py \
    --model_name $MODEL_NAME \
    --tensor_parallel_size 2 \
    --data_dir full_feedback \
    --interactive \
    --overwrite