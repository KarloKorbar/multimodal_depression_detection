import torch
from tqdm import tqdm

from training.trainer import BaseTrainer


class AudioRNNTrainer(BaseTrainer):
    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(batch_X)
            loss = self.criterion(output, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return train_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        progress_bar = tqdm(val_loader, desc="Validation")

        with torch.no_grad():
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                val_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return val_loss / len(val_loader)
