#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops,sc-loprio
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-output/run_model_%j.out
#SBATCH --error=slurm-output/run_model_%j.err
#SBATCH --constraint=[ampere|hopper]

source ~/.zshrc
cd ~/vlm-tg-context

# install conda if needed, create vtc environment if needed, activate vtc
source scripts/set_up_uv.sh

MODEL_NAME=$1
EXTRA_ARGS=$2

vllm serve $MODEL_NAME --host 0.0.0.0 --port 8000 &

python scripts/call_lm.py \
    --model_name $MODEL_NAME \
    $EXTRA_ARGS