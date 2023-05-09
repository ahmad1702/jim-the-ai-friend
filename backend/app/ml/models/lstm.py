import torch.nn as nn


# Encoder Long-Short Term Memory (LSTM)
class LSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, n_layers=1, dropout=0):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.LSTM = nn.LSTM(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())
        # Forward pass through GRU
        outputs, hidden = self.LSTM(packed, hidden)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Return output and final hidden state
        return outputs, hidden
