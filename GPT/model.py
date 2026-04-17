import torch
from torch import Tensor, nn
from Transformer import TransformerBlock


class Onyx(nn.Module):
    """
    Onyx GPT model implementation
    
    A transformer-based language model designed for autoregressive text generation.
    This model follows the GPT (Generative Pre-trained Transformer) architecture
    with multiple transformer blocks, embeddings, and a final output layer.
    
    Attributes:
        vocab_size (int): Size of the vocabulary
        model_dim (int): Dimensionality of the model embeddings
        context_size (int): Maximum context window size
        token_embedding (nn.Embedding): Token embedding layer
        positional_embedding (nn.Embedding): Positional embedding layer
        transformer_blocks (nn.ModuleList): List of transformer blocks
        output_layer (nn.Linear): Final output layer mapping to vocabulary
    """
    
    def __init__(self, vocab_size:int, model_dim:int = 256, num_heads:int = 4, 
                 num_layers:int = 4, ff_dim:int = 1024, context_size:int = 64, 
                 dropout:float = 0.1) -> None:
        """
        Initialize the Onyx GPT model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            model_dim (int): Dimensionality of the model embeddings
            num_heads (int): Number of attention heads in each transformer block
            num_layers (int): Number of transformer blocks
            ff_dim (int): Dimensionality of the feed-forward network
            context_size (int): Maximum context window size
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Store model configuration
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.context_size = context_size
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, model_dim)  # Token embeddings
        self.positional_embedding = nn.Embedding(context_size, model_dim)  # Positional embeddings
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, ff_dim, dropout) 
            for layer in range(num_layers)
        ])
        
        # Output layer (maps from model dim to vocabulary size)
        self.output_layer = nn.Linear(model_dim, vocab_size)
        
    def forward(self, X:Tensor) -> Tensor:
        """
        Forward pass of the Onyx GPT model.
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Tensor: Output logits of shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len = X.shape
        
        # Get token embeddings
        token_embeds = self.token_embedding(X)
        
        # Generate positional embeddings
        positions = torch.arange(seq_len, device=X.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.positional_embedding(positions)
        
        # Combine token and positional embeddings
        X = token_embeds + pos_embeds
        
        # Pass through each transformer block
        for block in self.transformer_blocks:
            X = block(X)
        
        # Generate output logits
        logits = self.output_layer(X)
        
        return logits

model = Onyx(22000)  
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")