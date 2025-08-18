#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=sc-loprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-output/vtc_%j.out
#SBATCH --error=slurm-output/vtc_%j.err
#SBATCH --constraint=[80G|141G]

source ~/.zshrc
cd ~/vlm-tg-context

# install conda if needed, create vtc environment if needed, activate vtc
source scripts/set_up_conda.sh

MODEL_NAME="moonshotai/Kimi-VL-A3B-Instruct"

python scripts/call_lm.py \
    --model $MODEL_NAME
