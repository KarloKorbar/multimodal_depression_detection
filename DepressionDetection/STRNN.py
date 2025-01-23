# STRNN as provided by GPT (look into this):

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and Preprocess Data
# Assume FacialFeatures and BinaryOutput are already loaded as Pandas DataFrames
# FacialFeatures is multi-indexed by ID and TIMESTAMP
# BinaryOutput is indexed by ID

# Align data by ID
aligned_ids = FacialFeatures.index.get_level_values('ID').unique().intersection(BinaryOutput.index)
FacialFeatures = FacialFeatures.loc[aligned_ids]
BinaryOutput = BinaryOutput.loc[aligned_ids]

# Normalize features per ID
def normalize_features(group):
    scaler = StandardScaler()
    group[:] = scaler.fit_transform(group)
    return group

FacialFeatures = FacialFeatures.groupby('ID').apply(normalize_features)

# Create input sequences (X) and labels (y)
X = FacialFeatures.values.reshape(len(aligned_ids), -1, FacialFeatures.shape[1])  # [n_subjects, timesteps, features]
y = BinaryOutput.values

# Step 2: Define STRNN Model
class STRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(STRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # rnn_out: [batch_size, timesteps, hidden_dim]
        out = self.fc(rnn_out[:, -1, :])  # Take last hidden state
        return self.activation(out)

# Step 3: Train-Test Split Using TimeSeriesSplit
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
splits = list(tscv.split(X))

# Step 4: Training Pipeline
input_dim = FacialFeatures.shape[1]
hidden_dim = 64
output_dim = 1
num_layers = 2
learning_rate = 0.001
batch_size = 16
num_epochs = 20

model = STRNN(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train and Validate Model
for train_idx, val_idx in splits:
    # Prepare data loaders
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                 torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                val_loss += criterion(outputs, y_batch).item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# Step 5: Evaluate Final Model on Test Data (if available)
# Here you would repeat the evaluation process on an unseen test set
# Assuming X_test and y_test are prepared similarly

# Example:
# test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model.eval()
# y_pred = []
# with torch.no_grad():
#     for X_batch, _ in test_loader:
#         outputs = model(X_batch).squeeze()
#         y_pred.extend(outputs.numpy())

# auc = roc_auc_score(y_test, y_pred)
# print(f"Test AUC: {auc:.4f}")
# print(classification_report(y_test, (np.array(y_pred) > 0.5).astype(int)))

