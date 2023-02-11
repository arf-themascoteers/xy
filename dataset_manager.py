import torch
import colour
import pandas as pd
import os
import math
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def create_dataset():
    source = "data.csv"
    X = torch.linspace(0,1,100).reshape(-1,2)
    y = torch.max(X,dim=1).values
    y = y.reshape(-1,1)
    all = torch.concat((X, y), dim=1)
    columns = ["x", "y"]
    df = pd.DataFrame(data=all, columns=columns)
    df.to_csv(source, index=False)


def get_dataset():
    source = "data.csv"
    create_dataset()
    return pd.read_csv(source).to_numpy()


if __name__ == "__main__":
    d = get_dataset()
    train_data, test_data = train_test_split(get_dataset(), random_state=2)

    train_x = torch.tensor(train_data[:,0:-1], dtype=torch.float32)
    train_y = torch.tensor(train_data[:,-1], dtype=torch.float32)

    test_x = torch.tensor(test_data[:,0:-1], dtype=torch.float32)
    test_y = torch.tensor(test_data[:,-1], dtype=torch.float32)

    plt.scatter(train_x.squeeze(), train_y)
    plt.show()

    plt.scatter(test_x.squeeze(), test_y)
    plt.show()
