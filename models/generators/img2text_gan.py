import torch
import torch.nn as nn

class Img2TextGenerator(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(True),
            nn.Conv2d(128,256,4,2,1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
