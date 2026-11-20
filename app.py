
import os
import streamlit as st
from personal_ai_model import get_ai_response


st.set_page_config(
    page_title="Yumi AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Yumi AI")
st.write(
    "Hello! I am Yumi AI. "
    "You can ask me anything in Chinese or English."
)

# Get Groq API key from Streamlit Secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.environ.get("GROQ_API_KEY", "")

if not api_key:
    st.error(
        "Groq API key not found. "
        "Please add GROQ_API_KEY to Streamlit Secrets."
    )
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("About Yumi AI")
    st.write("Powered by Groq")
    st.write("Supports Chinese and English.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input(
    "Ask Yumi AI anything..."
)

if question:
    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Yumi AI is thinking..."):
            try:
                answer = get_ai_response(
                    st.session_state.messages,
                    api_key
                )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(
                    "Sorry, I could not connect to the AI. "
                    "Please check your Groq settings. "
                    f"Error: {e}"
                )
