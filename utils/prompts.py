from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def get_prompt_template_from_messages(
    messages: List[Dict[str, str]]
) -> ChatPromptTemplate:

    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(SystemMessage(message["content"]))
        elif message["role"] == "user":
            formatted_messages.append(HumanMessage(message["content"]))
        elif message["role"] == "assistant":
            formatted_messages.append(AIMessage(message["content"]))

    for x in formatted_messages:
        print(x)

    return ChatPromptTemplate.from_messages(formatted_messages)
