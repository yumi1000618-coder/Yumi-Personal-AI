
import json
import os
from groq import Groq


MODEL_NAME = "openai/gpt-oss-20b"


def load_dataset():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "dataset.json"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_ai_response(chat_history, api_key):
    client = Groq(api_key=api_key)

    dataset = load_dataset()

    personal_information = json.dumps(
        dataset,
        ensure_ascii=False,
        indent=2
    )

    system_prompt = f"""
You are Yumi AI, a friendly and helpful personal AI assistant.

You can answer general questions like a normal AI assistant.
You can understand Chinese and English.

Here is Yumi's personal information dataset:
{personal_information}

Rules:
1. If the user asks about Yumi, use the dataset above.
2. Never invent personal information about Yumi.
3. If the dataset does not contain the answer, say that the information
   is not available in Yumi's dataset.
4. For general questions, answer using your general knowledge.
5. Reply in the same language as the user's latest message.
6. If the user asks multiple questions, answer every question separately.
7. Remember the conversation history when answering follow-up questions.
8. Be friendly, clear, and helpful.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Keep the recent conversation
    for message in chat_history[-20:]:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=2048
    )

    return completion.choices[0].message.content
