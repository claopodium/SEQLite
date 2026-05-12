import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(8,64,3,padding=1)
        self.conv2 = nn.Conv1d(64,128,3,padding=1)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):

        # (B,L,C) → (B,C,L)
        x = x.permute(0,2,1)


        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)

        return self.head(x).squeeze(-1)