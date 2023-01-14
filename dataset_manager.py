import torch
import colour
import pandas as pd
import os


def create_dataset():
    source = "data.csv"
    X = torch.rand((1000,3))
    y = 3*X[:,0] + 2 * X[:,1] + 0.6
    y = y.reshape(-1,1)
    all = torch.concat((X, y), dim=1)
    columns = ["x1", "x2", "x3", "y"]
    df = pd.DataFrame(data=all, columns=columns)
    df.to_csv(source, index=False)


def get_dataset():
    source = "data.csv"
    if not os.path.exists(source):
        create_dataset()
    return pd.read_csv(source).to_numpy()


if __name__ == "__main__":
    d = get_dataset()