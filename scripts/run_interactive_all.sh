MODELS=("gpt-5.2" "claude-4-5-sonnet")
API_BASES=("https://api.openai.com/v1" "https://api.anthropic.com/v1")
ARGS="--interactive"
for i in "${!MODELS[@]}"; do
    sbatch run_model_api.sh "${MODELS[$i]}" $ARGS --api_base "${API_BASES[$i]}"
done