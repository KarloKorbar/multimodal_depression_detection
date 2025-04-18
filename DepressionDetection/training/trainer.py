import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


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
