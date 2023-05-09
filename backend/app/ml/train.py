import numpy as np
import pandas as pd
import os
import random
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import nltk
from torch.utils.data import Dataset, DataLoader
import numpy as np
from nltk.stem.porter import PorterStemmer
import time

from .utils import (
    bag_of_words,
    all_words,
    tags,
    device,
    xy,
    model_filename,
)
from .accum_model import get_model
from .models.neural_net import NeuralNet


# Converts Seconds to Minutes and seconds. Better ux that way during training time
def seconds_to_minutes(duration_seconds):
    # convert duration to minutes and seconds
    minutes, seconds = divmod(duration_seconds, 60)

    # format the duration as "X m Ys"
    duration_str = f"{minutes} m {int(round(seconds))}s"

    return duration_str


def train_transformer_model():
    # hyperparameters
    transformer_hyperparameters = {
        "batch_size": 64,  # how many independent sequences will we process in parallel?
        "block_size": 256,  # what is the maximum context length for predictions?
        "max_iters": 15000,
        "eval_interval": 500,
        "learning_rate": 3e-4,
        "eval_iters": 200,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "dropout": 0.2,
        "model_file_name": "data.pt",
        "model_file_path": f"cached_model_states/data.pt",
    }
    torch.manual_seed(1337)

    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print("Data file loaded")

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    base_encode = {ch: i for i, ch in enumerate(chars)}
    base_decode = {i: ch for i, ch in enumerate(chars)}
    encode_func = lambda s: [base_encode[c] for c in s]

    # Train and test splits
    data = torch.tensor(encode_func(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # recieve a batch of data
    def obtain_group(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(
            len(data) - transformer_hyperparameters["block_size"], (batch_size,)
        )
        x = torch.stack(
            [data[i : i + transformer_hyperparameters["block_size"]] for i in ix]
        )
        y = torch.stack(
            [
                data[i + 1 : i + transformer_hyperparameters["block_size"] + 1]
                for i in ix
            ]
        )
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def loss_calculation():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(transformer_hyperparameters["eval_iters"])
            for k in range(transformer_hyperparameters["eval_iters"]):
                X, Y = obtain_group(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    class Head(nn.Module):
        """one head of self-attention"""

        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(
                transformer_hyperparameters["n_embd"], head_size, bias=False
            )
            self.query = nn.Linear(
                transformer_hyperparameters["n_embd"], head_size, bias=False
            )
            self.value = nn.Linear(
                transformer_hyperparameters["n_embd"], head_size, bias=False
            )
            self.register_buffer(
                "tril",
                torch.tril(
                    torch.ones(
                        transformer_hyperparameters["block_size"],
                        transformer_hyperparameters["block_size"],
                    )
                ),
            )

            self.dropout = nn.Dropout(transformer_hyperparameters["dropout"])

        def forward(self, x):
            # input of size (batch, time-step, channels)
            # output of size (batch, time-step, head size)
            B, T, C = x.shape
            k = self.key(x)  # (B,T,hs)
            q = self.query(x)  # (B,T,hs)
            # compute attention scores ("affinities")
            wei = (
                q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
            )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
            wei = F.softmax(wei, dim=-1)  # (B, T, T)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(x)  # (B,T,hs)
            out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
            return out

    class MultiHAttn(nn.Module):
        """multiple heads of self-attention in parallel"""

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(
                head_size * num_heads, transformer_hyperparameters["n_embd"]
            )
            self.dropout = nn.Dropout(transformer_hyperparameters["dropout"])

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedFwrd(nn.Module):
        """a simple linear layer followed by a non-linearity"""

        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(transformer_hyperparameters["dropout"]),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """Transformer block: communication followed by computation"""

        def __init__(self, n_embd, n_head):
            # n_embd: embedding dimension, n_head: the number of heads we'd like
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHAttn(n_head, head_size)
            self.ffwd = FeedFwrd(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class NLPModel(nn.Module):
        def __init__(self):
            super().__init__()
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(
                vocab_size, transformer_hyperparameters["n_embd"]
            )
            self.position_embedding_table = nn.Embedding(
                transformer_hyperparameters["block_size"],
                transformer_hyperparameters["n_embd"],
            )
            self.blocks = nn.Sequential(
                *[
                    Block(
                        transformer_hyperparameters["n_embd"],
                        n_head=transformer_hyperparameters["n_head"],
                    )
                    for _ in range(transformer_hyperparameters["n_layer"])
                ]
            )
            self.ln_f = nn.LayerNorm(
                transformer_hyperparameters["n_embd"]
            )  # final layer norm
            self.lm_head = nn.Linear(transformer_hyperparameters["n_embd"], vocab_size)

            # better init, not covered in the original GPT video, but important, will cover in followup video
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape

            # idx and targets are both (B,T) tensor of integers
            tok_emb = self.token_embedding_table(idx)  # (B,T,C)
            pos_emb = self.position_embedding_table(
                torch.arange(T, device=device)
            )  # (T,C)
            x = tok_emb + pos_emb  # (B,T,C)
            x = self.blocks(x)  # (B,T,C)
            x = self.ln_f(x)  # (B,T,C)
            logits = self.lm_head(x)  # (B,T,vocab_size)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -transformer_hyperparameters["block_size"] :]
                # get the predictions
                logits, loss = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :]  # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            return idx

    print("PyTorch version:", torch.__version__)
    model = NLPModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # Check if saved model exists
    if os.path.isdir("cache") and os.path.isfile("cache/jim.pth"):
        print("Loading cached model")
        m.load_state_dict(torch.load("cache/jim.pth"))
    else:
        print("Training model")
        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        training_start_time = time.time()
        for iter in range(transformer_hyperparameters["max_iters"]):
            # every once in a while evaluate the loss on train and val sets
            if (
                iter % transformer_hyperparameters["eval_interval"] == 0
                or iter == transformer_hyperparameters["max_iters"] - 1
            ):
                losses = loss_calculation()
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {seconds_to_minutes(time.time() - training_start_time)}"
                )

            # sample a batch of data
            xb, yb = obtain_group("train")

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print("Saving model")
        torch.save(m.state_dict(), "cache/jim.pth")

    print("Model is loaded")


# create training data
X_train = []
y_train = []
for pattern_sentence, tag in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

if os.path.isfile("cached_model_states/data.pth"):
    train_transformer_model()

model = get_model(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


print(f"final loss: {loss.item():.4f}")


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

torch.save(data, model_filename)

print(f"training complete. file saved to {model_filename}")
