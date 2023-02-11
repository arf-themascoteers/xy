from torch import nn


class MyMachine(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size,3),
            nn.ReLU(),
            nn.Linear(3,1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x