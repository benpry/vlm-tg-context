MODELS=("Qwen/Qwen3-VL-32B-Instruct" "allenai/Molmo2-8B" "google/gemma-3-27b-it")

ARGS="--interactive"

for model in "${MODELS[@]}"; do
    sbatch run_model.sh $model $ARGS
done