import torch
import colour
import pandas as pd
import os


def create_dataset():
    source = "data.csv"
    X = torch.linspace(0,1,1000).reshape(-1,1)
    y = torch.sin(X[:,0]*10)
    y = y.reshape(-1,1)
    all = torch.concat((X, y), dim=1)
    columns = ["x", "y"]
    df = pd.DataFrame(data=all, columns=columns)
    df.to_csv(source, index=False)


def get_dataset():
    source = "data.csv"
    #if not os.path.exists(source):
    create_dataset()
    return pd.read_csv(source).to_numpy()


if __name__ == "__main__":
    d = get_dataset()