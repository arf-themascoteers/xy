import torch
from torch import nn
from sklearn.metrics import r2_score


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


def get_dataset():
        X = torch.rand((1000,2))
        y = (X[:,0]**2) + (X[:,0] * X[:,1])
        return X, y


def train():
    model = MyMachine()
    model.train()
    X, y = get_dataset()
    NUM_EPOCHS = 800
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss(reduction='mean')

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(X)
        y_pred = y_pred.reshape(-1)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch}, Loss:{loss.item()}')
    torch.save(model.state_dict(), 'model.h5')
    return model


def test():
    model = MyMachine()
    model.load_state_dict(torch.load("model.h5"))
    model.eval()
    X, y = get_dataset()

    with torch.no_grad():
        y_pred = model(X)
        print(r2_score(y, y_pred))