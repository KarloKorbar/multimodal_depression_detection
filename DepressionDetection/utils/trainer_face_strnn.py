import torch

from utils.trainer import BaseTrainer


class FaceSTRNNTrainer(BaseTrainer):
    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            output, spatial_weights, temporal_weights = self.model(batch_X)
            loss = self.criterion(output, batch_y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                output, spatial_weights, temporal_weights = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                val_loss += loss.item()

        return val_loss / len(val_loader)
