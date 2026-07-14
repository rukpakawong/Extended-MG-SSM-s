import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, optimizer, criterion, device=None):
        """
        Initialize the Trainer.

        Args:
            model: The model to be trained
            optimizer: The optimizer to use for training
            criterion: The loss function
            device: 'cuda', 'mps', or 'cpu'. If None, it auto-detects.
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Push the model to selected device
        self.model = self.model.to(self.device)

        # Lists to keep track of loss over time (useful for plotting later)
        self.train_losses = []
        self.val_losses = []

    def train_step(self, train_loader):
        """
        Runs one epoch of training
        """
        # Set model to training mode
        self.model.train() 
        running_loss = 0.0

        for inputs, targets in train_loader:

            # 1. Push data to the correct device (GPU/CPU)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # 2. Clear old gradients
            self.optimizer.zero_grad()

            # 3. Forward pass
            predictions = self.model(inputs)

            # 4. Compute loss
            loss = self.criterion(predictions, targets)

            # 5. Backward pass
            loss.backward()

            # 6. Update weights
            self.optimizer.step()

            running_loss += loss.item()*inputs.size(0)

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss
    
    def validate_step(self, val_loader):
        """
        Runs one epoch of validation (no learning).
        """
        # Set model to evaluation mode
        self.model.eval()
        running_loss = 0.0

        # Disable gradient calculation for validation to save memory and speed up computation
        with torch.no_grad():
            for inputs, targets in val_loader:

                # 1. Push data to the correct device (GPU/CPU)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 2. Forward    
                predictions = self.model(inputs)

                # 3. Compute loss
                loss = self.criterion(predictions, targets)

                running_loss += loss.item()*inputs.size(0)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs):
        """
        The main loop that orchestrates training and validation
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):

            # Run training and validation for this epoch
            train_loss = self.train_step(train_loader)
            val_loss = self.validate_step(val_loader)

            # Store the metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        print("Training complete!")
        return self.train_losses, self.val_losses