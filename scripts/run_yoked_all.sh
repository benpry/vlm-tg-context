# MODELS=("meta-llama/Llama-3.2-11B-Vision-Instruct" "Qwen/Qwen2.5-VL-32B-Instruct" "Qwen/Qwen3-VL-32B-Instruct" "google/gemma-3-27b-it" "moonshotai/Kimi-VL-A3B-Instruct")
MODELS=("Qwen/Qwen3-VL-32B-Instruct")
ARGS="--yoked"

for model in ${MODELS[@]}; do
    sbatch run_model.sh $model $ARGS
done
