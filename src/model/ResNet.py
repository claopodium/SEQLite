import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):

    def __init__(self, encode):

        super().__init__()
        if encode == "ONEHOT":
            wide = 20
        elif encode == "VHSE":
            wide = 8

        self.conv1 = nn.Conv1d(wide,64,3,padding=1)
        self.conv2 = nn.Conv1d(64,128,3,padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        self.res = nn.Conv1d(8, 128, 1)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )

    def forward(self,x):

        # (B,L,C) → (B,C,L)
        x = x.permute(0,2,1)

        shortcut = self.res(x)

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x) + shortcut)
        x = self.bn2(x)
        x = self.pool(x).squeeze(-1)

        return self.head(x).squeeze(-1)