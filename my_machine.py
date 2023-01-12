from torch import nn


class MyMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3,10),
            nn.LeakyReLU(),
            nn.Linear(20,10),
            nn.LeakyReLU(),
            nn.Linear(10,3)
        )

    def forward(self, x):
        x = self.fc(x)
        return x