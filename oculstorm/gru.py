"""A small GRU that predicts the next hour's change in Dst from a window of history."""
import torch
from torch import nn


class DstGRU(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, last = self.gru(x)
        return self.head(last[-1]).squeeze(-1)
