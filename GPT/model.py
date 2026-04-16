import torch
from torch import Tensor, nn
from Transformer import TransformerBlock


class Onyx(nn.Module):
    def __init__(self, vocab_size:int, model_dim:int = 256, num_heads:int = 4, num_layers:int = 4,ff_dim = 1024, context_size:int = 64, dropout:float = 0.1) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.context_size = context_size
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_embedding = nn.Embedding(context_size, model_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(model_dim, num_heads, ff_dim, dropout) for layer in range(num_layers)])
        
        # Output layer
        self.output_layer = nn.Linear(model_dim, vocab_size)
        
    def forward(self, X:Tensor) -> Tensor:
        batch_size, seq_len = X.shape
        
        #Get Embeddings
        token_embeds = self.token_embedding(X)
        positions = torch.arange(seq_len, device=X.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embeds = self.positional_embedding(positions)
        
        # Combine the embeddings
        X = token_embeds + pos_embeds
        
        # Pass through transformer block
        for block in self.transformer_blocks:
            X = block(X)
        
        # Output logits
        logits = self.output_layer(X)
        
        return logits

model = Onyx(22000)  
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
        
        
        
        
