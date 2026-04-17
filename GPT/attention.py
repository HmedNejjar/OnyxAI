from torch import Tensor, nn
import torch
import torch.nn.functional as funct


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    
    Implements the multi-head attention mechanism used in transformer models.
    This allows the model to attend to different parts of the input sequence
    simultaneously through multiple "heads" that focus on different aspects.
    
    Attributes:
        model_dim (int): Dimensionality of the model embeddings
        num_heads (int): Number of attention heads
        head_dim (int): Dimensionality of each attention head
        q_linear (nn.Linear): Linear layer for query projections
        k_linear (nn.Linear): Linear layer for key projections
        v_linear (nn.Linear): Linear layer for value projections
        out_linear (nn.Linear): Linear layer for output projections
    """
    
    def __init__(self, model_dim:int, num_heads:int) -> None:
        """
        Initialize the MultiHeadAttention module.
        
        Args:
            model_dim (int): Dimensionality of the model embeddings
            num_heads (int): Number of attention heads
        """
        super().__init__()
        
        # Store configuration
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # Dimensionality per head
        
        # Ensure model dimension is divisible by number of heads
        assert model_dim % num_heads == 0, "Model dimension must be divisible by number of heads"
        
        # Linear layers for query, key, and value projections
        self.q_linear = nn.Linear(model_dim, model_dim)  # Query projection
        self.k_linear = nn.Linear(model_dim, model_dim)  # Key projection
        self.v_linear = nn.Linear(model_dim, model_dim)  # Value projection
        
        # Output linear layer
        self.out_linear = nn.Linear(model_dim, model_dim)  # Final projection
    
    def forward(self, X:Tensor) -> Tensor:
        """
        Forward pass of the MultiHeadAttention module.
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, sequence_length, model_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, model_dim)
        """
        batch_size, seq_len, _ = X.shape
        
        # Compute query, key, and value projections
        q = self.q_linear(X)  # (batch_size, seq_len, model_dim)
        k = self.k_linear(X)  # (batch_size, seq_len, model_dim)
        v = self.v_linear(X)  # (batch_size, seq_len, model_dim)
        
        # Reshape for multi-head attention
        # Split into multiple heads and transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # Dot product of query and key, scaled by square root of head dimension
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Create causal mask to prevent attending to future tokens
        # This ensures the model only looks at past and current tokens
        mask = torch.tril(torch.ones(seq_len, seq_len, device=X.device)).unsqueeze(0).unsqueeze(0)
        score = score.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights using softmax
        attention = funct.softmax(score, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attention, v)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back to original shape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        
        # Apply output linear layer
        out = self.out_linear(out)
        
        return out