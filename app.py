
import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# Load the saved AI model
with open("personal_ai_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

vocab = bundle["vocab"]
answers = bundle["answers"]
word_to_id = {word: i for i, word in enumerate(vocab)}

# Rebuild the neural network
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

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def get_answer(question):
    words = tokenize(question)
    known_words = [w for w in words if w in word_to_id]

    if not known_words:
        return "Sorry, I don't understand that question yet."

    x = torch.zeros(1, len(vocab))

    for word in known_words:
        x[0, word_to_id[word]] = 1

    with torch.no_grad():
        prediction = model(x).argmax(dim=1).item()

    return answers[prediction]

# Streamlit interface
st.set_page_config(page_title="Yumi AI", page_icon="🤖")
st.title("🤖 Yumi AI")
st.write("Welcome to my personal AI! Ask me about my information.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask me something...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    answer = get_answer(question)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()
