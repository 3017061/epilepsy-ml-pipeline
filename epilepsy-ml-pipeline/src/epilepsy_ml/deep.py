from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        z = self.net(x)        # (B,64,1)
        z = z.squeeze(-1)      # (B,64)
        return self.fc(z)

class LSTM1D(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64, n_layers: int = 1, n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden, n_classes)

    def forward(self, x):      # x: (B,C,T)
        x = x.transpose(1,2)   # (B,T,C)
        o, _ = self.lstm(x)
        h = o[:,-1]            # last time step
        return self.fc(h)

class CNNLSTM(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):      # x: (B,C,T)
        z = self.cnn(x)        # (B,32,T/4)
        z = z.transpose(1,2)   # (B,T/4,32)
        o, _ = self.lstm(z)
        h = o[:,-1]
        return self.fc(h)

@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 64
    device: str = "cpu"

def train_torch(model: nn.Module, X, y, cfg: TrainConfig):
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device(cfg.device)
    model.to(device)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
    return model
