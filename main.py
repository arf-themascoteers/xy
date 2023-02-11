import torch
from torch import nn
from sklearn.metrics import r2_score
import colour
import pandas as pd
from my_machine import MyMachine
from dataset_manager import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def train(X, y):
    model = MyMachine(X.shape[1])
    model.train()
    NUM_EPOCHS = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
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


def test(X, y):
    model = MyMachine(X.shape[1])
    model.load_state_dict(torch.load("model.h5"))
    model.eval()

    with torch.no_grad():
        y_pred = model(X)
        print(r2_score(y, y_pred))


def do_linear(train_x, train_y, test_x, test_y):
    model = LinearRegression()
    model = model.fit(train_x, train_y)
    r2 = model.score(test_x, test_y)
    print("LR",r2)

if __name__ == "__main__":
    train_data, test_data = train_test_split(get_dataset(), random_state=1, test_size=0.4)

    train_x = torch.tensor(train_data[:,0:-1], dtype=torch.float32)
    train_y = torch.tensor(train_data[:,-1], dtype=torch.float32)
    train(train_x, train_y)

    test_x = torch.tensor(test_data[:,0:-1], dtype=torch.float32)
    test_y = torch.tensor(test_data[:,-1], dtype=torch.float32)
    test(test_x, test_y)

    # plt.scatter(train_x.squeeze(), train_y)
    # plt.show()
    #
    # plt.scatter(test_x.squeeze(), test_y)
    # plt.show()

    #do_linear(train_x, train_y, test_x, test_y)
