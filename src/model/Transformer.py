import torch
import torch.nn as nn

class Trans(nn.Module):
    def __init__(self, embedding, d_m = 64, nhead = 4):
        super().__init__()

        max_len = 128

        if embedding == "ONEHOT":
            self.embedding = nn.Linear(20, d_m)
        elif embedding == "VHSE":
            self.embedding = nn.Linear(8, d_m)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_m,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2
        )

        self.pos_embedding = nn.Parameter(
        torch.randn(1, max_len, d_m) * 0.02
        )  

        self.attn_pool = nn.Linear(d_m,1)
        self.fc1 = nn.Linear(d_m, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc1(x)