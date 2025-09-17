import warnings
from ast import literal_eval


def get_image_token(model_name, include_image: bool = True):
    """
    Get the image token for a given model.
    """
    if not include_image:
        return ""

    model_name = model_name.lower()
    if "gemma" in model_name:
        return "<start_of_image>"
    elif "llama" in model_name:
        return "<|image|>"
    elif "qwen" in model_name:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    elif "kimi" in model_name:
        return "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_messages(system_prompt, chat_prompt, include_image, grid_image, model_name):
    messages = []
    if "gemma" in model_name.lower():
        # gemma models don't have a system role, so we add the instruction to the first user message
        messages = [*chat_prompt]
        if include_image:
            messages[0]["content"] = [
                {"type": "image", "image": grid_image},
                {"type": "text", "text": f"{system_prompt}\n{messages[0]['content']}"},
            ]
        else:
            messages[0]["content"] = f"{system_prompt}\n{messages[0]['content']}"
    elif "llama" in model_name.lower():
        # llama models don't take images in the system prompt, so we add the image to the first user message
        messages = [
            {"role": "system", "content": system_prompt},
            *chat_prompt,
        ]
        messages[1]["content"] = [
            {"type": "image", "image": grid_image},
            {"type": "text", "text": messages[1]["content"]},
        ]
    else:
        # otherwise the image goes in the system prompt
        if include_image:
            system_content = [
                {"type": "image", "image": grid_image},
                {"type": "text", "text": system_prompt},
            ]
        else:
            system_content = system_prompt

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            *chat_prompt,
        ]

    return messages


def get_user_message(messages):
    """
    Get the user message from a list of messages.
    """
    if not isinstance(messages, list):
        return ""

    user_message = ""
    for message in messages:
        if "text" not in message:
            warnings.warn(f"Message {message} is missing 'text' field.")
        else:
            user_message += f"{message['role']}: {message['text']}\n"

    return user_message


def add_user_message(chat_messages: list, user_message: str):
    if chat_messages and chat_messages[-1]["role"] == "user":
        chat_messages[-1]["content"] += "\n\n" + user_message
    else:
        chat_messages.append({"role": "user", "content": user_message})

    return chat_messages


def preprocess_messages(row):
    """
    Turn a row of the dataframe into a list of messages for the chat model.
    """
    chat_messages = []
    message_history = row["message_history"]
    if isinstance(message_history, str):
        message_history = literal_eval(message_history.replace("nan", "''"))
    else:
        message_history = []

    selection_history = row["selection_history"]
    if isinstance(selection_history, str):
        selection_history = literal_eval(selection_history)
    else:
        selection_history = []

    correctness_history = row["correctness_history"]
    if isinstance(correctness_history, str):
        correctness_history = literal_eval(correctness_history)
    else:
        correctness_history = []

    for messages, selection, correctness in zip(
        message_history, selection_history, correctness_history
    ):
        user_message = get_user_message(messages)
        chat_messages = add_user_message(chat_messages, user_message)
        chat_messages.append({"role": "assistant", "content": selection})
        chat_messages = add_user_message(
            chat_messages, "Correct." if correctness else "Incorrect."
        )

    this_trial_messages = row["message"]
    if not isinstance(this_trial_messages, str):
        chat_messages = add_user_message(chat_messages, "describer: \n")
    else:
        this_trial_messages = literal_eval(this_trial_messages.replace("nan", "''"))
        chat_messages = add_user_message(
            chat_messages, get_user_message(this_trial_messages)
        )

    return chat_messages


def get_logprobs_from_outputs(outputs, choice_tokens):
    """
    Get the log probabilities of the choice tokens from the model outputs.
    """
    all_choice_logprobs = []
    for output in outputs:
        choice_logprobs = {}
        all_choice_logprobs.append(choice_logprobs)
        logprobs = output.outputs[0].logprobs[0].values()
        for logprob in logprobs:
            if logprob.decoded_token.strip() in choice_tokens:
                choice_logprobs[logprob.decoded_token.strip()] = logprob.logprob
            if len(choice_logprobs) == len(choice_tokens):
                break

        if len(choice_logprobs) < len(choice_tokens):
            warnings.warn("Not all choice tokens found in top logprobs.")

    return all_choice_logprobs
