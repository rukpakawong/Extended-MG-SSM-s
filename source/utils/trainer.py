import os
import torch
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, model, optimizer, criterion, device=None, save_dir="checkpoints"):
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

        # Setup directory for saving the model
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Push the model to selected device
        self.model = self.model.to(self.device)

        # Lists to keep track of loss over time (useful for plotting later)
        self.train_losses = []
        self.val_losses = []

    def train_step(self, train_loader, max_grad_norm=1.0):
        """
        Runs one epoch of training with gradient clipping (default: 1.0).
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

            # 5.1 Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

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

    def train(self, train_loader, val_loader, epochs, patience=15, label=None):
        """
        The main loop that orchestrates training and validation with 
        early stopping (default: 15) and checkpointing.
        """
        print(f"Starting training on device: {self.device}")

        best_val_loss = float('inf')
        early_stopping_counter = 0
        save_path = os.path.join(self.save_dir, f"best_{self.model.__class__.__name__}_{label}.pth")

        pbar = tqdm(range(epochs), desc="Training Model")
        for epoch in pbar:

            # Run training and validation for this epoch
            train_loss = self.train_step(train_loader)
            val_loss = self.validate_step(val_loader)

            # Store the metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(self.model.state_dict(), save_path)

                # Print progress
                pbar.set_postfix_str(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Saved Best Model)")
            else:
                early_stopping_counter += 1

                # Print progress
                pbar.set_postfix_str(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {early_stopping_counter}/{patience}")

            # Check for early stopping
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete! Best validation loss: {best_val_loss:.4f}. Model saved to {save_path}")

        # Load the best weights back into the model before returning
        self.model.load_state_dict(torch.load(save_path))

        return self.train_losses, self.val_losses