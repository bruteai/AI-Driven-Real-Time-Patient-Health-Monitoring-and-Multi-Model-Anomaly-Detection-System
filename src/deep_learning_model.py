import torch
import torch.nn as nn

class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.layer = nn.Linear(5, 1)
    def forward(self, x):
        return self.layer(x)
