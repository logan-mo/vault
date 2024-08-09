from langchain_core.prompts import ChatPromptTemplate


def get_prompt_template_from_messages(messages):
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(("system", message["content"]))
        elif message["role"] == "user":
            formatted_messages.append(("human", message["content"]))
        elif message["role"] == "assistant":
            formatted_messages.append(("ai", message["content"]))

    return ChatPromptTemplate.from_messages(formatted_messages)
