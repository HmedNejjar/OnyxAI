import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from DataLoader import GPTDataset
from save_params import save_params, load_params
from model import Onyx
from tqdm import tqdm
from pathlib import Path
import sys

PARENT_FOLDER = Path(r'G:\Projects\Python\OnyxAI')

sys.path.insert(2, str(PARENT_FOLDER))
from Tokenizer.BPE import BPE


VOCAB_SIZE = 22000
MODEL_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 1024
CONTEXT_SIZE = 64
DROPOUT = 0.1
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
LOG_PERIOD = 10
PATIENCE = 20  # Early stopping patience
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = PARENT_FOLDER / Path(r'Tokenizer/merges.json')
TRAIN_PATH = PARENT_FOLDER / Path(r'Corpus/TinyStoriesV2-train.txt')
VAL_PATH = PARENT_FOLDER / Path(r'Corpus/TinyStoriesV2-valid.txt')
MODEL_SAVE_PATH = Path('best_model.pth')  # Path for the best model


def load_corpus(path: Path | str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
    
def load_tokenizer(path:Path | str) -> BPE:
    tokenizer = BPE(VOCAB_SIZE)
    print(f"Loading existing tokenizer from: {path}\n")
    tokenizer.merges = tokenizer.load(path)
    return tokenizer

def train() -> None:
    # Load utilities
    print("Loading corpus and tokenizer...\n")
    train_corpus = load_corpus(TRAIN_PATH)
    val_corpus = load_corpus(VAL_PATH)
    tokenizer = load_tokenizer(TOKENIZER)
    print(f"Training corpus size: {len(train_corpus):,} characters\n")
    print(f"Validation corpus size: {len(val_corpus):,} characters\n")
    
    # Create Datasets
    print("Creating datasets...\n")
    train_dataset = GPTDataset(train_corpus, tokenizer, CONTEXT_SIZE)
    val_dataset = GPTDataset(val_corpus, tokenizer, CONTEXT_SIZE)
    
    # Create Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    #Initiate model
    print("Initializing model...\n")
    model = Onyx(VOCAB_SIZE, MODEL_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM, CONTEXT_SIZE, DROPOUT).to(DEVICE)
    
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
            
            #Calculate Loss
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), target.view(-1))
            
            #Backprop and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            #Log training progress
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
    """Run validation and return average loss"""
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