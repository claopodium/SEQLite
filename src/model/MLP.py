import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, seq_length, embedding, task):
        super().__init__()

        if embedding == "ONEHOT":
            wide = 20
        elif embedding == "VHSE":
            wide = 8

        self.fc1 = nn.Linear(seq_length * wide, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

        self.task = task # category or regression

    def forward(self, x):
        """
        x: (batch_size, L, A)
        """
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = torch.relu(self.fc3(x))

        y_hat = self.fc4(x)

        if self.task == "Binary":
            y_hat = torch.sigmoid(y_hat)
        
        return y_hat