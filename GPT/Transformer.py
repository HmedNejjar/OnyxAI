from attention import MultiHeadAttention
from FFN import FeedForward
from torch import Tensor, nn

class TransformerBlock(nn.Module):
    """
    Transformer block implementation
    
    A single transformer block that consists of multi-head attention,
    layer normalization, and a feed-forward network. This block is the
    building block of the transformer architecture used in the Onyx model.
    
    Attributes:
        attention (MultiHeadAttention): Multi-head attention mechanism
        feed_forward (FeedForward): Feed-forward neural network
        layer_norm1 (nn.LayerNorm): Layer normalization after attention
        layer_norm2 (nn.LayerNorm): Layer normalization after feed-forward
        dropout (nn.Dropout): Dropout layer for regularization
    """
    
    def __init__(self, model_dim:int, num_heads:int, ff_dim:int, dropout:float) -> None:
        """
        Initialize the TransformerBlock.
        
        Args:
            model_dim (int): Dimensionality of the model embeddings
            num_heads (int): Number of attention heads
            ff_dim (int): Dimensionality of the feed-forward network
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(model_dim, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(model_dim, ff_dim)
        
        # Layer normalization layers
        self.layer_norm1 = nn.LayerNorm(model_dim)  # After attention
        self.layer_norm2 = nn.LayerNorm(model_dim)  # After feed-forward
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X:Tensor) -> Tensor:
        """
        Forward pass of the TransformerBlock.
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, sequence_length, model_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, model_dim)
        """
        # Multi-head attention with residual connection
        attn_out = self.attention(X)
        X = self.layer_norm1(X + self.dropout(attn_out))
        
        # Feed-forward network with residual connection
        ff_out = self.feed_forward(X)
        X = self.layer_norm2(X + self.dropout(ff_out))
        
        return X