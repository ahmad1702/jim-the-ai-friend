import torch.nn as nn


# Encoder MLP
class MLP(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # As MLP is an RNN Model that has at least 3 Fully-Connected Layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        output = self.embedding(input_seq)

        # 3 fully connected layer, the output of the first layer becomes the input of the second layer and same goes to layer 3
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(output, input_lengths.cpu())

        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]

        # Return output and final hidden state
        return outputs, hidden
