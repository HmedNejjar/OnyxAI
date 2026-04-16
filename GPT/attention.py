from torch import Tensor, nn
import torch
import torch.nn.functional as funct

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim:int, num_heads:int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim / num_heads
        
        # Linear Layers Q, K, V
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        
        # Output linear layer
        self.out_linear = nn.Linear(model_dim, model_dim)
    
    def forward(self, X:Tensor) -> Tensor:
        batch_size, seq_len = X.shape
        
        # Compute Q, K, V
        q, k, v = self.q_linear(X), self.k_linear(X), self.v_linear(X)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v= v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # Compute attention score parameter
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=X.device)).unsqueeze(0).unsqueeze(0)
        score = score.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention score
        attention = funct.softmax(score, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        
        # Apply output linear layer
        out = self.out_linear(out)
        
        return out