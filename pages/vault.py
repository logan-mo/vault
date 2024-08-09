import openai
import streamlit as st

from utils.generator import get_context
from utils.llms import LLMFactory, EmbeddingsFactory
from utils.prompts import get_prompt_template_from_messages

openai.api_key = st.secrets["OPENAI_API_KEY"]


def get_system_prompt():
    return """
    Du bist VAULT.AI, ein hilfreicher Assistent, der die Workshop-Teilnehmenden bei der Exploration von dem VAULT zum Thema GPT im Unternehmen unterstützt.

    Du verwendest ausschließlich die Informationen aus dem Kontext <context>, um die Antworten zu geben. 

    Dabei nutzt du im Detail die Kontextinformationen im <context> und achtest darauf, nur die wichtigsten Punkte zu erklären und besonders gut auf die Frage des Nutzers zu antworten.

    Am Anfang begrüßt du den Nutzer (kannst immer duzen!) und erklärst kurz über deine Funktionalitäten: 

    "

    Hallo, ich bin VAULT.AI <-- erkläre hier 

    Ich kann zum Beispiel folgende Fragen beantworten: 

    - was ist InhouseGPT
    - was ist In-Context Learning
    - was denkst du zu BloombergGPT

    Mein Ziel ist, dich zu unterstützen, so dass du eine KI im eigenen Unternehmen entwickeln kannst!

    ***
    """


def show():
    st.title("Talk to VAULT")

    # Initialize the chat messages history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": get_system_prompt()}]

    # Prompt for user input
    if prompt := st.chat_input(placeholder="Ask questions about InhouseGPT"):
        augmented_prompt = get_context(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "user", "content": augmented_prompt})

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "system" or message["role"] == "context":
            continue
        if message["content"].startswith("<context>"):
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # if "results" in message:
            #     st.dataframe(message["results"])

    # If last message is not from assistant, we need to generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()

            # Create a stream for chat completions
            # stream = openai.chat.completions.create(
            #     model="gpt-4o-2024-05-13", # gpt-3.5-turbo is cheaper
            #     messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            #     stream=True,
            # )

            llm = LLMFactory.create_openai_llm(
                model="gpt-4o-mini",
                api_key=st.secrets["OPENAI_API_KEY"],
            )

            stream = llm.stream(
                get_prompt_template_from_messages(st.session_state.messages)
            )
            for chunk in stream:
                if chunk is not None:
                    response += chunk
                    resp_container.markdown(response)

            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    show()
