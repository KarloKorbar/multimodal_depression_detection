import torch
from tqdm import tqdm

from training.trainer import BaseTrainer


class MultimodalFusionTrainer(BaseTrainer):
    def __init__(
        self, model, criterion, optimizer, scheduler, device, early_stopping_patience=7
    ):
        super().__init__(model, criterion, optimizer, scheduler, device)
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_text, batch_audio, batch_face, batch_y in progress_bar:
            # Move all inputs to device
            batch_text = batch_text.to(self.device)
            batch_audio = batch_audio.to(self.device)
            batch_face = batch_face.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_text, batch_audio, batch_face)
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_train_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0
        progress_bar = tqdm(val_loader, desc="Validation")

        with torch.no_grad():
            for batch_text, batch_audio, batch_face, batch_y in progress_bar:
                # Move all inputs to device
                batch_text = batch_text.to(self.device)
                batch_audio = batch_audio.to(self.device)
                batch_face = batch_face.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_text, batch_audio, batch_face)
                val_loss = self.criterion(outputs, batch_y)
                total_val_loss += val_loss.item()

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{val_loss.item():.4f}"})

        return total_val_loss / len(val_loader)

    def train(self, train_loader, val_loader, n_epochs):
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            # Validation
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch": epoch,
                    },
                    "best_multimodal_model.pth",
                )
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        return train_losses, val_losses
