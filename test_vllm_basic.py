#!/usr/bin/env python3

from vllm import LLM, SamplingParams


def test_vllm_logprobs():
    # Initialize the model
    llm = LLM(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    )  # You can change this to your preferred model

    # Example prompts to test
    prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "The weather today is",
        "Machine learning is",
    ]

    # Configure sampling parameters for logprobs
    sampling_params = SamplingParams(
        max_tokens=1,  # Generate only one token
        logprobs=20,  # Get top 20 logprobs
        temperature=0.0,  # Deterministic for consistency
    )

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # Print results and examine structure
    print("=== vLLM Output Structure Analysis ===\n")

    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: '{prompts[i]}'")
        print(f"Generated text: '{output.outputs[0].text}'")
        print(f"Output type: {type(output)}")
        print(f"Output attributes: {dir(output)}")

        # Examine the first output in detail
        first_output = output.outputs[0]
        print(f"First output type: {type(first_output)}")
        print(f"First output attributes: {dir(first_output)}")

        # Examine logprobs structure
        if hasattr(first_output, "logprobs") and first_output.logprobs:
            print(f"Logprobs type: {type(first_output.logprobs)}")
            print(f"Logprobs length: {len(first_output.logprobs)}")

            if first_output.logprobs:
                first_logprob = first_output.logprobs[0]
                print(f"First logprob entry type: {type(first_logprob)}")
                print(f"First logprob entry attributes: {dir(first_logprob)}")

                # Print the actual logprobs data
                print("Logprobs data structure:")
                if hasattr(first_logprob, "__dict__"):
                    for key, value in first_logprob.__dict__.items():
                        print(f"  {key}: {type(value)} = {value}")

                # Print top tokens and their probabilities
                print("Top 20 tokens and logprobs:")
                if hasattr(first_logprob, "items") and callable(first_logprob.items):
                    for token, logprob in first_logprob.items():
                        print(f"  '{token}': {logprob}")
                elif hasattr(first_logprob, "keys") and hasattr(
                    first_logprob, "values"
                ):
                    for token, logprob in zip(
                        first_logprob.keys(), first_logprob.values()
                    ):
                        print(f"  '{token}': {logprob}")

        print("-" * 60)


if __name__ == "__main__":
    test_vllm_logprobs()
