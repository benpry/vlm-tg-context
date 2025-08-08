import warnings
from ast import literal_eval

from sglang.lang.chat_template import get_chat_template


def get_image_token(model_name):
    """
    Get the image token for a given model.
    """
    if "gemma" in model_name:
        return "<start_of_image>"
    elif "llama" in model_name:
        return "<|image|>"
    elif "idefics" in model_name:
        return "<image>"
    elif "Qwen" in model_name:
        return "<|vision_bos|><|image_pad|><|vision_eos|>"
    else:
        raise ValueError(f"Model {model_name} not supported")


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

    target_history = row["target_history"]
    if isinstance(target_history, str):
        target_history = literal_eval(row["target_history"])
    else:
        target_history = []

    for messages, target in zip(message_history, target_history):
        user_message = get_user_message(messages)
        chat_messages.append({"role": "user", "content": user_message})
        chat_messages.append({"role": "assistant", "content": target})

    this_trial_messages = row["message"]
    if not isinstance(this_trial_messages, str):
        chat_messages.append({"role": "user", "content": "describer: \n"})
    else:
        this_trial_messages = literal_eval(this_trial_messages.replace("nan", "''"))
        chat_messages.append(
            {"role": "user", "content": get_user_message(this_trial_messages)}
        )

    return chat_messages


def get_sgl_chat_template(model_name):
    """
    Get the chat template for a given model.
    """
    if "Qwen" in model_name:
        return get_chat_template("qwen2-vl")
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_logprobs_from_outputs(outputs, choice_tokens, choice_token_ids):
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
