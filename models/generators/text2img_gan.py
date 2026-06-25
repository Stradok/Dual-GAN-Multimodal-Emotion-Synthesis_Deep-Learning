import torch
import torch.nn as nn
from .film import FiLMModulation

class Text2ImageGenerator(nn.Module):
    def __init__(self, emb_dim=512, ngf=64, output_nc=3):
        super().__init__()
        # Initial dense projection
        self.fc = nn.Linear(emb_dim, ngf*8*8*8)
        self.ngf = ngf

        # Upsampling conv layers
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_nc, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.ngf*8, 8, 8)
        x = self.up(x)
        return x
