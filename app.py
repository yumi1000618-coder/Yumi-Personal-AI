
import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# -----------------------------
# Load AI Model
# -----------------------------
with open("personal_ai_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

vocab = bundle["vocab"]
answers = bundle["answers"]
word_to_id = {word: i for i, word in enumerate(vocab)}

class PersonalAI(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.network(x)


model = PersonalAI(len(vocab))
model.load_state_dict(bundle["state_dict"])
model.eval()


# -----------------------------
# Personal Information
# -----------------------------
personal_info = {
    "name": "Yumi",
    "shoe_size": "23.5 cm",
    "height": "158 cm",
    "weight": "50 kg"
}


# -----------------------------
# Tokenize
# -----------------------------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# -----------------------------
# AI Answer Function
# -----------------------------
def get_answer(question):
    q = question.lower().strip()

    # Name
    if any(x in q for x in [
        "my name", "who am i", "call me",
        "what is my name", "introduce yourself"
    ]):
        return "My name is Yumi!"

    # Shoe size
    if any(x in q for x in [
        "shoe", "shoes", "footwear"
    ]):
        return "My shoe size is 23.5 cm."

    # Height
    if any(x in q for x in [
        "height", "tall"
    ]):
        return "My height is 158 cm."

    # Weight
    if any(x in q for x in [
        "weight", "weigh"
    ]):
        return "My weight is 50 kg."

    # All information
    if any(x in q for x in [
        "about me", "everything",
        "my information", "all about me"
    ]):
        return (
            "Here is what I know about myself:\n\n"
            "- Name: Yumi\n"
            "- Shoe size: 23.5 cm\n"
            "- Height: 158 cm\n"
            "- Weight: 50 kg"
        )

    # Original trained AI model
    words = tokenize(question)
    known_words = [w for w in words if w in word_to_id]

    if not known_words:
        return "Sorry, I don't understand that question yet."

    x = torch.zeros(1, len(vocab))

    for word in known_words:
        x[0, word_to_id[word]] = 1

    with torch.no_grad():
        prediction = model(x).argmax(dim=1).item()

    if prediction >= len(answers):
        return "Sorry, I don't know the answer yet."

    return answers[prediction]


# -----------------------------
# Streamlit Interface
# -----------------------------
st.set_page_config(
    page_title="Yumi AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Yumi AI")
st.subheader("Welcome to my Personal AI!")

st.write(
    "Hello! I am Yumi's personal AI. "
    "Ask me about my personal information!"
)

# Sidebar
with st.sidebar:
    st.title("About Yumi")
    st.write("Name: Yumi")
    st.write("Shoe size: 23.5 cm")
    st.write("Height: 158 cm")
    st.write("Weight: 50 kg")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input("Ask me something...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    answer = get_answer(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()
