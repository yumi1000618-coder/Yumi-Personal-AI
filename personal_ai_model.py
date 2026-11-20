
import json
import os
import re


def load_dataset():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "dataset.json"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def search_dataset(question):
    dataset = load_dataset()
    question = question.lower().strip()

    matched_answers = []

    # Split multiple questions
    questions = re.split(r"[?？\n]+", question)

    for q in questions:
        q = q.strip()

        if not q:
            continue

        best_match = None
        best_score = 0

        for item in dataset:
            text = (
                item["question"] + " " +
                item["answer"] + " " +
                item.get("category", "")
            ).lower()

            words = re.findall(r"\b[a-z0-9]+\b", q)
            score = sum(
                1 for word in words
                if len(word) > 2 and word in text
            )

            if score > best_score:
                best_score = score
                best_match = item

        if best_match and best_score >= 1:
            matched_answers.append(best_match["answer"])

    if matched_answers:
        return "\n\n".join(dict.fromkeys(matched_answers))

    return "Sorry, I cannot find this information in my dataset."
