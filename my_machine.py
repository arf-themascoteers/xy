from torch import nn


class MyMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3,20),
            nn.LeakyReLU(),
            nn.Linear(20,5),
            nn.LeakyReLU(),
            nn.Linear(5,3)
        )

    def forward(self, x):
        x = self.fc(x)
        return x