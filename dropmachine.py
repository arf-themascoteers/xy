from torch import nn


class DropMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(3, 2)
        self.d = nn.Dropout()
        self.l2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        print(x.shape)
        x = self.d(x)
        print(x.shape)
        x = self.l2(x)
        print(x.shape)
        return x
