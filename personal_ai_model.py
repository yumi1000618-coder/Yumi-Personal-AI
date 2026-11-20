
import json
import os

def load_dataset():
    file_path = os.path.join(
        os.path.dirname(__file__),
        "dataset.json"
    )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def search_dataset(question):
    dataset = load_dataset()
    question = question.lower()

    best_match = None
    best_score = 0

    for item in dataset:
        text = (
            item["question"] + " " +
            item["answer"] + " " +
            item.get("category", "")
        ).lower()

        words = question.split()
        score = sum(1 for word in words if word in text)

        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score > 0:
        return best_match["answer"]

    return "Sorry, I cannot find this information in my dataset."

