MODELS = ("Llama-3.2-11B-Vision-Instruct", "Qwen2.5-VL-32B-Instruct", "gemma-3-27b-it", "Kimi-VL-A3B-Instruct")

ARGS = "--data_dir full_feedback --api_base http://localhost:8000 --interactive"

for model in ${MODELS[@]}; do
    sbatch run_model.sh $model $ARGS
done