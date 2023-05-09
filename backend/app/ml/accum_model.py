import torch
import torch.nn as nn

from .models.bigram import BigramLanguageModel
from .models.lstm import LSTM

from .utils import device, all_words

# Hyper-parameters
hidden_size = 256
vocab_size = len(all_words)

# bigram model
bigram = BigramLanguageModel(vocab_size)
bigram = bigram.to(device)

# LSTM
lstm = LSTM(256, vocab_size)

# A collection of a few models that can be used to predict the next


# LSTM() returns tuple of (tensor, (recurrent state))
class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


import torch
import torch.nn as nn


# the main model that includes an LSTM, Luong Attention, MLP, and RNN.
class AccumulativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AccumulativeModel, self).__init__()

        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size)

        # Luong Attention
        self.attention = LuongAttention(hidden_size)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

        # RNN
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # LSTM
        lstm_output, _ = self.lstm(x)

        # Luong Attention
        attention_output, _ = self.attention(lstm_output)

        # MLP
        mlp_output = self.mlp(attention_output)

        # RNN
        rnn_output, _ = self.rnn(x)

        return mlp_output, rnn_output


# The LuongAttention class is a separate module used within the main model to implement the attention mechanism.
class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()

        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs):
        seq_len = inputs.size(1)

        # Calculate attention scores
        attn_scores = torch.tanh(self.attn(inputs))
        attn_weights = torch.softmax(self.v(attn_scores), dim=1)

        # Apply attention weights
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), inputs)

        return attn_applied, attn_weights


def get_model(input_size, hidden_size, output_size):
    return AccumulativeModel(input_size, hidden_size, output_size)
