import torch
import torch.nn as nn

class Img2TextDiscriminator(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
