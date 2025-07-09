import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        attention_weights = self.attention(x)  # (batch, seq_len, 1)
        attended_features = x * attention_weights
        return attended_features, attention_weights


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hidden_states):
        # hidden_states shape: (batch, seq_len, hidden_dim)
        attention_weights = self.attention(hidden_states)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(
            hidden_states * attention_weights, dim=1
        )  # (batch, hidden_dim)
        # Ensure context is always 2D (batch_size, hidden_dim)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        return context, attention_weights


class FaceSTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(FaceSTRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Spatial attention
        self.spatial_attention = SpatialAttention(input_size)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_size * 2
        )  # *2 for bidirectional

        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_embedding=False):
        x, spatial_weights = self.spatial_attention(x)
        lstm_out, _ = self.lstm(x)
        context, temporal_weights = self.temporal_attention(lstm_out)
        embedding = torch.relu(self.fc1(context))
        embedding = self.batch_norm(embedding)
        embedding = self.dropout(embedding)
        logits = self.fc2(embedding)
        if return_embedding:
            return embedding  # shape: (batch, hidden_size)
        return logits, spatial_weights, temporal_weights
