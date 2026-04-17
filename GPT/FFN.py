from torch import Tensor, nn


class FeedForward(nn.Module):
    """
    Feed-forward neural network module
    
    Implements the feed-forward network used in transformer blocks.
    This consists of two linear layers with a LeakyReLU activation in between.
    The feed-forward network allows the model to process the attention output
    and capture complex non-linear relationships in the data.
    
    Attributes:
        linear1 (nn.Linear): First linear layer (expansion)
        linear2 (nn.Linear): Second linear layer (contraction)
        ReLU (nn.LeakyReLU): LeakyReLU activation function
    """
    
    def __init__(self, model_dim:int, ff_dim: int) -> None:
        """
        Initialize the FeedForward module.
        
        Args:
            model_dim (int): Dimensionality of the model embeddings
            ff_dim (int): Dimensionality of the hidden layer
        """
        super().__init__()
        
        # First linear layer (expands dimension)
        self.linear1 = nn.Linear(model_dim, ff_dim)
        
        # Second linear layer (contracts back to model dimension)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        
        # Activation function
        self.ReLU = nn.LeakyReLU()
        
    def forward(self, X:Tensor) -> Tensor:
        """
        Forward pass of the FeedForward module.
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, sequence_length, model_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, model_dim)
        """
        # Pass through first linear layer
        X = self.linear1(X)
        
        # Apply activation function
        X = self.ReLU(X)
        
        # Pass through second linear layer
        X = self.linear2(X)
        
        return X