from torch import Tensor, nn

class FeedForward(nn.Module):
    def __init__(self, model_dim:int, ff_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.ReLU = nn.LeakyReLU()
        
    def forward(self, X:Tensor) -> Tensor:
        X = self.linear1(X)
        X = self.ReLU(X)
        X = self.linear2(X)
        
        return X