import torch
import torch.nn as nn
import torch.nn.functional as F

class SLP(nn.Module):
    def __init__(self, seq_length, embedding:str):
        super().__init__()
        if embedding == "ONEHOT":
            wide = 20
        elif embedding == "VHSE":
            wide = 8
        
        self.theta = nn.Parameter(
            torch.randn(seq_length, wide) * 0.001
        )
        # (L, A)

        self.bias = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        """
        x: (batch_size, L, A)
        """
        y_hat = torch.sum(x * self.theta, dim=(1, 2)) + self.bias
        
        # y_hat = torch.sigmoid(y_hat)
        
        return y_hat
