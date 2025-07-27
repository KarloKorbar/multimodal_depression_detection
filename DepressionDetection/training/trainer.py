import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from pathlib import Path


class BaseTrainer(ABC):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        early_stopping_patience=7,
        checkpoint_dir="checkpoints",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

    @abstractmethod
    def train_epoch(self, train_loader):
        pass

    @abstractmethod
    def validate(self, val_loader):
        pass

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pth")

        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.learning_rates = checkpoint["learning_rates"]
        return checkpoint["epoch"]

    def train(self, train_loader, val_loader, n_epochs, resume_from=None):
        start_epoch = 0

        # Resume training if checkpoint provided
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, n_epochs):
            # Clear output in Jupyter notebook
            from IPython.display import clear_output
            clear_output(wait=True)

            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate tracking
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Early stopping check
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                is_best = True
            else:
                self.early_stopping_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)

            # Print progress
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
            )

            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return self.train_losses, self.val_losses


# Plot training and validation loss curves.
def plot_training_curves(
    train_losses, val_losses, title="Training and Validation Loss"
):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# Save model state and related information.
def save_model(model, scaler, input_size, best_params, save_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_state_dict": scaler,
            "input_size": input_size,
            "best_params": best_params,
        },
        save_path,
    )


# Load a saved model and its related information.
def load_model(model_class, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)

    best_params = checkpoint["best_params"].copy()
    [best_params.pop(k, None) for k in ("learning_rate", "weight_decay")]
    model = model_class(input_size=checkpoint["input_size"], **best_params).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint["scaler_state_dict"]
    # input_size = checkpoint['input_size']
    # best_params = checkpoint['best_params']

    return model, scaler
