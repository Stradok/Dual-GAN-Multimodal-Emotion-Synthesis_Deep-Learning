import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def __call__(self, preds, target_is_real=True):
        target = torch.ones_like(preds) if target_is_real else torch.zeros_like(preds)
        return self.criterion(preds, target)
