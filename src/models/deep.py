
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x)
