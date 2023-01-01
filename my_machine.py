import torch
import torch.nn as nn


class MyMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,10),
            nn.LeakyReLU(),
            nn.Linear(10,5),
            nn.LeakyReLU(),
            nn.Linear(5,1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x