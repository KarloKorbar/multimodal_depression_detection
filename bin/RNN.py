# RNN as provided by GPT (look into it):

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset

# Load your datasets (replace with actual data loading)
# features = pd.read_csv("features.csv")
# output = pd.read_csv("output.csv")

# Example structure for merging datasets
# features.reset_index(inplace=True)  # Bring ID and TIMESTAMP into columns
# output.reset_index(inplace=True)   # Bring ID into a column
# data = features.merge(output, on="ID", how="inner")

# Define custom Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.X = torch.tensor(data.iloc[:, 2:-1].values, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define RNN model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        output = self.fc(hidden[-1])
        return self.sigmoid(output)

# Preprocessing (example with placeholder data)
# scaler = StandardScaler()
# data.iloc[:, 2:-1] = scaler.fit_transform(data.iloc[:, 2:-1])

# TimeSeriesSplit for data splitting
tscv = TimeSeriesSplit(n_splits=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training pipeline
for train_idx, test_idx in tscv.split(data):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_dataset = TimeSeriesDataset(train_data)
    test_dataset = TimeSeriesDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = RNNClassifier(input_size=data.shape[1]-2, hidden_size=64, num_layers=2).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(20):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

