import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# TODO: think about extracting all the models into their own files to make it cleaner

class BaseTrainer(ABC):
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    @abstractmethod
    def train_epoch(self, train_loader):
        pass

    @abstractmethod
    def validate(self, val_loader):
        pass

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

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return train_losses, val_losses


class AudioRNNTrainer(BaseTrainer):
    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(batch_X)
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
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                val_loss += loss.item()

        return val_loss / len(val_loader)


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


class MultimodalFusionTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, scheduler, device, early_stopping_patience=7):
        super().__init__(model, criterion, optimizer, scheduler, device)
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_train_loss = 0

        for batch_text, batch_audio, batch_face, batch_y in train_loader:
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

        return total_train_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_text, batch_audio, batch_face, batch_y in val_loader:
                # Move all inputs to device
                batch_text = batch_text.to(self.device)
                batch_audio = batch_audio.to(self.device)
                batch_face = batch_face.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_text, batch_audio, batch_face)
                val_loss = self.criterion(outputs, batch_y)
                total_val_loss += val_loss.item()

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
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch,
                }, 'best_multimodal_model.pth')
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return train_losses, val_losses


# Plot training and validation loss curves.
def plot_training_curves(train_losses, val_losses, title='Training and Validation Loss'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Save model state and related information.
def save_model(model, scaler, save_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state_dict': scaler,
        # 'input_size': input_size,
        # 'best_params': best_params
    }, save_path)


# Load a saved model and its related information.
def load_model(model_class, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)

    model = model_class(
        input_size=checkpoint['input_size'],
        **checkpoint['best_params']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler_state_dict']
    # input_size = checkpoint['input_size']
    # best_params = checkpoint['best_params']

    return model, scaler
