import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from DataLoader import GPTDataset
from save_params import save_params, load_params
from model import Onyx
from tqdm import tqdm
from pathlib import Path
import sys

"""
Training script for the Onyx GPT model

This script handles the training process for the Onyx GPT model, including:
- Loading pre-tokenized datasets
- Initializing the model
- Training with AdamW optimizer
- Validation and early stopping
- Saving the best model
"""

PARENT_FOLDER = Path(r'G:\Projects\Python\OnyxAI')

sys.path.insert(2, str(PARENT_FOLDER))
from Tokenizer.BPE import BPE


# Model and training configuration
VOCAB_SIZE = 22000      # Size of the token vocabulary
MODEL_DIM = 256         # Dimensionality of model embeddings
NUM_HEADS = 4           # Number of attention heads
NUM_LAYERS = 4          # Number of transformer layers
FF_DIM = 1024           # Dimensionality of feed-forward network
CONTEXT_SIZE = 64       # Maximum context window size
DROPOUT = 0.1           # Dropout probability
BATCH_SIZE = 32         # Batch size for training
LR = 1e-4               # Learning rate
EPOCHS = 100            # Maximum number of epochs
LOG_PERIOD = 10         # Log training progress every N batches
PATIENCE = 20           # Early stopping patience
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device to use

# File paths
TOKENIZER = PARENT_FOLDER / Path(r'Tokenizer/bpe_30k_vocab.json')  # Path to tokenizer
TRAIN_PATH = PARENT_FOLDER / Path(r'Corpus/tokenized_train.npy')  # Path to training tokens
VAL_PATH = PARENT_FOLDER / Path(r'Corpus/tokenized_valid.npy')  # Path to validation tokens
MODEL_SAVE_PATH = Path('best_model.pth')  # Path to save the best model

    
def load_tokenizer(path: Path | str) -> BPE:
    """
    Load the BPE tokenizer from disk.
    
    Args:
        path (Path | str): Path to the tokenizer file
        
    Returns:
        BPE: Loaded BPE tokenizer instance
    """
    tokenizer = BPE(VOCAB_SIZE)
    print(f"Loading existing tokenizer from: {path}\n")
    tokenizer.load(path)
    return tokenizer

def train() -> None:
    """
    Train the Onyx GPT model.
    """
    # Load utilities
    print("Loading tokenizer...\n")
    tokenizer = load_tokenizer(TOKENIZER)
    print(f"Tokenizer loaded successfully\n")
    
    # Create Datasets
    print("Creating datasets...\n")
    train_dataset = GPTDataset(TRAIN_PATH, CONTEXT_SIZE)
    val_dataset = GPTDataset(VAL_PATH, CONTEXT_SIZE)
    
    print(f"Training dataset size: {len(train_dataset):,} samples\n")
    print(f"Validation dataset size: {len(val_dataset):,} samples\n")
    
    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("Initializing model...\n")
    model = Onyx(VOCAB_SIZE, MODEL_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, CONTEXT_SIZE, DROPOUT).to(DEVICE)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters\n")
    
    # Load pre-trained model if available
    if MODEL_SAVE_PATH.exists():
        print(f"Loading pre-trained model from {MODEL_SAVE_PATH}...")
        model_state = load_params(MODEL_SAVE_PATH)
        model.load_state_dict(model_state)
        print("Pre-trained model loaded successfully\n")
    
    # Configure Optimizer and Loss function
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    # Track best validation loss and early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training Loop
    print("Starting training...\n")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training")
        
        for i, (context, target) in enumerate(train_iterator):
            # Move data to device
            context, target = context.to(DEVICE), target.to(DEVICE)
            
            # Forward Pass
            logits = model(context)
            
            # Calculate Loss
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), target.view(-1))
            
            # Backpropagation and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Log training progress
            if (i + 1) % LOG_PERIOD == 0:
                avg_loss = train_loss / (i + 1)
                train_iterator.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        
        # Run validation
        val_loss = validate(model, val_loader, loss_fn, epoch)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_params(model, MODEL_SAVE_PATH)
            print(f"  --> Saved best model (val loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement ({epochs_no_improve}/{PATIENCE})")
        
        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to {MODEL_SAVE_PATH}")
    
def validate(model: Onyx, val_dataloader: DataLoader, loss_fn: nn.Module, epoch: int) -> float:
    """
    Run validation and return average loss.
    
    Args:
        model (Onyx): Onyx model to validate
        val_dataloader (DataLoader): Validation data loader
        loss_fn (nn.Module): Loss function to use
        epoch (int): Current epoch number
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        valid_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation", leave=False)
        for context, target in valid_iterator:
            context, target = context.to(DEVICE), target.to(DEVICE)
            
            # Forward Pass
            logits = model(context)
            
            # Calculate Loss
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), target.view(-1))
            val_loss += loss.item()
            
    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_dataloader)
    return avg_val_loss
    
if __name__ == "__main__":
    train()