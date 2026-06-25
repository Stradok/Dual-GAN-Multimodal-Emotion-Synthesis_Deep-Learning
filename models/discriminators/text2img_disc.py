import torch
import torch.nn as nn

class Text2ImageDiscriminator(nn.Module):
    def __init__(self, emb_dim=512, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1,1)
