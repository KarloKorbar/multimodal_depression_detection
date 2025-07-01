import torch
import torch.nn as nn


class AudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(AudioRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Output layers
        self.fc1 = nn.Linear(
            hidden_size, hidden_size // 2
        )  # Dynamic size based on hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 2)  # 2 classes for binary classification

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length=1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # shape: (batch_size, hidden_size)

        # Final classification layers
        out = self.dropout(torch.relu(self.fc1(context_vector)))
        out = self.fc2(out)
        return out
