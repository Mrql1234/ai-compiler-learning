from __future__ import annotations

import torch


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
