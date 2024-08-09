import streamlit as st

try:
    from pages import vault, data, settings
except FileNotFoundError as e:
    st.switch_page("pages/settings.py")
    from pages import vault, data, settings


def main():
    # Set the page config
    st.set_page_config(
        page_title="VAULT_APP - Explore InhouseGPT",
        page_icon=":speech_balloon:",
        layout="wide",
    )

    # Sidebar
    st.sidebar.image("logo.png", width=200)  # Replace with your logo file
    st.sidebar.title("VAULT_APP")
    st.sidebar.text("Explore InhouseGPT")

    # Define the pages
    pages = {
        "Vault": vault,
        "Data": data,
        "Settings": settings,
    }

    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Display the selected page with the session state
    pages[selection].show()

    # Hide "Made with Streamlit" and the burger icon
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
