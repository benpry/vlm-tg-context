from ast import literal_eval


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
        user_message += f"{message['role']}: {message['text']}\n"

    return user_message


def preprocess_messages(row):
    """
    Turn a row of the dataframe into a list of messages for the chat model.
    """
    chat_messages = []
    this_trial_messages = row["messages"]
    if not isinstance(this_trial_messages, str):
        chat_messages.append({"role": "user", "content": "describer: \n"})
    else:
        this_trial_messages = literal_eval(this_trial_messages.replace("nan", "''"))
        chat_messages.append(
            {"role": "user", "content": get_user_message(this_trial_messages)}
        )

    return chat_messages
