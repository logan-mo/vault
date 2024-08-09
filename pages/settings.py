import streamlit as st
from PIL import Image
import os


def secrets_file_exists() -> bool:
    return os.path.isfile(".streamlit/secrets.toml")


def initialize_secrets():
    with open(".streamlit/secrets.toml", "w") as f:
        f.write("")


def show():
    st.title("Settings")

    # # App Name
    # app_name = st.text_input("App Name", value=st.session_state.get("app_name", "My Streamlit App"))
    # if app_name != st.session_state.get("app_name"):
    #     st.session_state.app_name = app_name
    #     st.rerun()

    # Logo Upload
    uploaded_logo = st.file_uploader("Upload Logo", type=["png"])
    if uploaded_logo is not None:
        logo = Image.open(uploaded_logo)
        logo.save("logo.png")
        st.success("Logo updated successfully!")

    if not secrets_file_exists():
        st.info("No secrets file found. Creating a new one.")
        initialize_secrets()

    # API Keys
    openai_api_key = st.text_input(
        "OpenAI API Key", type="password", value=st.secrets.get("OPENAI_API_KEY", "")
    )
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        type="password",
        value=st.secrets.get("PINECONE_API_KEY", ""),
    )
    pinecone_index_name = st.text_input(
        "Pinecone Index Name", value=st.secrets.get("PINECONE_INDEX", "")
    )
    pinecone_cloud = st.text_input(
        "Pinecone Cloud", value=st.secrets.get("PINECONE_CLOUD", "aws")
    )
    pinecone_environment = st.text_input(
        "Pinecone Environment", value=st.secrets.get("PINECONE_ENVIRONMENT", "")
    )
    cohere_api_key = st.text_input(
        "Cohere API Key", type="password", value=st.secrets.get("COHERE_API_KEY", "")
    )
    huggingface_token = st.text_input(
        "Hugging Face Token",
        type="password",
        value=st.secrets.get("HUGGINGFACE_TOKEN", ""),
    )

    if st.button("Save Settings"):
        # In a real application, you'd want to securely store these keys
        # For this example, we'll just use the following code to update the secrets file
        secretsfile = ".streamlit/secrets.toml"
        with open(secretsfile, "w") as f:
            f.write(
                f'OPENAI_API_KEY="{openai_api_key}"\n'
                f'PINECONE_API_KEY="{pinecone_api_key}"\n'
                f'PINECONE_INDEX="{pinecone_index_name}"\n'
                f'PINECONE_CLOUD="{pinecone_cloud}"\n'
                f'PINECONE_ENVIRONMENT="{pinecone_environment}"\n'
                f'COHERE_API_KEY="{cohere_api_key}"\n'
                f'HUGGINGFACE_TOKEN="{huggingface_token}"\n'
            )


if __name__ == "__main__":
    show()
