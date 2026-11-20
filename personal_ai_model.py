
import json
import os
import re
from difflib import SequenceMatcher


def load_dataset():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "dataset.json"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def search_dataset(question):
    dataset = load_dataset()

    # Split the input into separate questions
    questions = re.split(r"[?？!\n]+", question)

    answers = []

    for q in questions:
        q = normalize(q)

        if not q:
            continue

        best_match = None
        best_score = 0

        for item in dataset:
            dataset_question = normalize(item["question"])

            score = SequenceMatcher(
                None, q, dataset_question
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = item

        if best_match and best_score >= 0.45:
            answers.append(best_match["answer"])
        else:
            answers.append(
                "Sorry, I cannot find this information in my dataset."
            )

    if answers:
        return "\n\n".join(answers)

    return "Please enter a question."
