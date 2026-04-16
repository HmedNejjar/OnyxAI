from attention import MultiHeadAttention
from FFN import FeedForward
from torch import Tensor, nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, model_dim:int, num_heads:int, ff_dim:int, dropout:float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = FeedForward(model_dim, ff_dim)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X:Tensor) -> Tensor:
        attn_out = self.attention(X)
        X = self.layer_norm1(X + self.dropout(attn_out))
        ff_out = self.feed_forward(X)
        X = self.layer_norm2(X + self.dropout(ff_out))
        
        return X