import os
from json import load
import torch
import torch.nn as nn
import json
import nltk
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Any

from .accum_model import get_model

from .utils import (
    tokenize,
    bag_of_words,
    all_words,
    tags,
    device,
    intents,
    model_filename,
)
from .models.neural_net import NeuralNet

with open("./app/ml/new_intents.json", "r") as f:
    intents = json.load(f)

if os.path.isfile(model_filename):
    data = torch.load(model_filename)
else:
    from .train import data as new_data

    data: dict[str, Any] = new_data

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = get_model(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


async def chat(input):
    try:
        sentence = input
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.10:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return {
                        "message": random.choice(intent["responses"]),
                    }
        else:
            return {
                "message": "I do not understand...",
            }
    except Exception as e:
        error = str(e)
        print(error)
        return {
            "error": "Something went wrong calling the 'chat_input' from the jupyter notebook"
        }
