import sys
import torch
from torch import nn
import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from model import BinaryClassifier, process_train_and_test, batchify
from tqdm import tqdm


def train_ann(train_data, max_input_length, batch_size=32):
    """
    Train an ANN model using memory-efficient batching with NumPy arrays.
    
    Args:
        X_train (np.ndarray): Training data of shape (num_samples, num_features).
        y_train (np.ndarray): Training labels of shape (num_samples,).
        max_input_length (int): Number of input features.
        batch_size (int): Number of samples per batch.
    """
    input_size = max_input_length
    model = BinaryClassifier(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create directories for saving checkpoints and models
    save_checkpoints_dir = os.path.join("data", "results", "ann", "checkpoints")
    save_models_dir = os.path.join("data", "results", "ann", "models")
    os.makedirs(save_checkpoints_dir, exist_ok=True)
    os.makedirs(save_models_dir, exist_ok=True)
    
    # Training loop
    num_epochs = 100
    epoch_progress = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_progress:
        epoch_loss = 0.0
        num_batches = 0
        
        # Create batches for this epoch
        for batch in batchify(train_data, batch_size=batch_size):
            # Convert NumPy arrays to PyTorch tensors
            x_batch, y_batch = batch[:, :-1], batch[:, -1]
            X_tensor = torch.tensor(x_batch, dtype=torch.float32)
            y_tensor = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1)  # Add channel dim
            
            # Forward pass
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Compute average loss
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
        epoch_progress.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(save_checkpoints_dir, f'checkpoint_{epoch+1}.pth'))
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_models_dir, 'model.pth'))

def test_ann(test_data, model):
    #TODO: Complete this function
    ...

if __name__ == "__main__":

    data_dir = '...'
    max_input_length = 512
    train, test = process_train_and_test(data_dir, max_input_length)
    train_ann(train, max_input_length, batch_size=16)
    # test_ann(X_test, y_test, max_input_length)
