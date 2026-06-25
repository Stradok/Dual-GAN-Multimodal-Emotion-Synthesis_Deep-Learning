import torch
import torch.nn as nn

class FiLMModulation(nn.Module):
    def __init__(self, emb_dim, num_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 512),  # project embedding to 512
            nn.ReLU(),
            nn.Linear(512, num_channels * 2)
        )

    def forward(self, x, z):
        params = self.fc(z)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta
