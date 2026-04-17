import torch
from pathlib import Path


def save_params(model, path: Path | str):
    """
    Save model parameters to disk.
    
    Args:
        model: PyTorch model to save
        path (Path | str): Path to save the model parameters
        
    Returns:
        Path: Actual path where the model was saved
    """
    model_path = Path(path)

    # Handle directory vs file path
    if model_path.suffix == '':
        model_dir = model_path
        model_path = model_dir / f"best_model.pth"
    else:
        model_dir = model_path.parent

    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    torch.save(obj=model.state_dict(), f=model_path)
    print(f"Model saved to: {model_path}")
    return model_path
    
def load_params(path: Path | str):
    """
    Load model parameters from disk.
    
    Args:
        path (Path | str): Path to the saved model parameters
        
    Returns:
        dict: Model state dictionary
    """
    print(f"Loading model from: {path}")
    return torch.load(path)

def freeze(model, train_transformer: bool = True, train_embeddings: bool = True, train_output: bool = True):
    """
    Freeze specific model components to control trainable parameters.
    
    Args:
        model: PyTorch model to freeze components of
        train_transformer (bool): Whether to train transformer blocks
        train_embeddings (bool): Whether to train embedding layers
        train_output (bool): Whether to train output layer
    """
    # Freeze embedding layers
    for param in model.token_embedding.parameters():
        param.requires_grad = train_embeddings
    
    for param in model.positional_embedding.parameters():
        param.requires_grad = train_embeddings
    
    # Freeze transformer blocks
    for block in model.transformer_blocks:
        for param in block.parameters():
            param.requires_grad = train_transformer
    
    # Freeze output layer
    for param in model.output_layer.parameters():
        param.requires_grad = train_output
    
    # Calculate trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")