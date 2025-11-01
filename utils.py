import os
import re
import string
import random
from random import shuffle
import json
import random
from prompt_template import PromptTemplate
from minutes_writer import *


def load_data(input_path):
    print()
    print("================================")
    print(f"Loading data from {input_path}.")
    print("================================")
    print()

    if input_path.endswith('.json'):
        with open(input_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    elif input_path.endswith('.jsonl'):
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
            data = data[0]['data']
    else:  # .txt
        data = open(input_path, 'r')

    return data


def save_json(obj, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print("================================")
    print(f"Saved at {file_path}.")
    print("================================")


def api_setup(model, api_key):
    if api_key == "":
        if "gpt" in model:
            with open("./openai_api_key.txt", 'r') as f:
                api_key = f.readline().strip()
        else:  # hf model
            api_key = api_key
            print("No API model used.")
    else:
        api_key = api_key

    return api_key


def normalize_text(text):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def prompt_split_msg(input_prompt):
    system_prompt, user_prompt = input_prompt.split("---")[0], input_prompt.split("---")[-1]

    contents = dict()
    contents["system"] = system_prompt
    contents["user"] = user_prompt

    messages = [
        {"role": "system",
         "content": contents["system"]},
        {"role": "user",
         "content": contents["user"]}
    ]

    return messages


def generate_choices_format(gold, choice_list):
    text = [gold.lower()] + choice_list
    shuffle(text)

    labels = ["A", "B", "C", "D"]

    answer_index = text.index(gold.lower())

    choices = {
        "text": text,
        "label": labels,
    }

    answer_key = labels[answer_index]

    return choices, answer_key
