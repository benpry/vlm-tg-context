#!/bin/zsh
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-output/eval_float32_%j.out
#SBATCH --error=slurm-output/eval_float32_%j.err


source ~/.zshrc

conda activate vtc

cd ~/vlm-tg-context

export LD_LIBRARY_PATH="/scr/benpry/conda/envs/vtc/lib64:/scr/benpry/conda/envs/vtc/lib:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-12.6"
export VLLM_USE_V1=0

python scripts/call_lm.py --overwrite