
import streamlit as st
from personal_ai_model import search_dataset

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


# Sidebar
with st.sidebar:
    st.title("About Yumi")
    st.write("Name: Yumi")
    st.write("Shoe size: 23.5 cm")
    st.write("Height: 158 cm")
    st.write("Weight: 50 kg")
    st.write("Favorite colors: White and Black 🤍🖤")
    st.write("Birthday: June 18")
    st.write("Favorite food: Matcha")
    st.write("Favorite subject: Physical Education")

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
