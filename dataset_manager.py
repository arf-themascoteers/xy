import torch
import colour
import pandas as pd
import os


def rgb_to_hsv(X):
    hsv_array = torch.zeros((X.shape[0],3), dtype = torch.float32)
    for i in range(X.shape[0]):
        hsv = colour.RGB_to_HSV([X[i][0].item(), X[i][1].item(), X[i][2].item()])
        hsv_array[i,0], hsv_array[i,1], hsv_array[i,2] = hsv[0], hsv[1], hsv[2]
    return hsv_array


def create_dataset():
    source = "data.csv"
    r = torch.linspace(0, 1, 10)
    g = torch.linspace(0, 1, 10)
    b = torch.linspace(0, 1, 10)
    X = torch.cartesian_prod(r,g,b)
    y = rgb_to_hsv(X)
    all = torch.concat((X, y), dim=1)
    columns = ["r", "g", "b", "h", "s", "v"]
    df = pd.DataFrame(data=all, columns=columns)
    df.to_csv(source, index=False)


def get_dataset():
    source = "data.csv"
    if not os.path.exists(source):
        create_dataset()
    data = pd.read_csv(source).to_numpy()
    X, y = data[:,0:3], data[:3:]
    return X, y


if __name__ == "__main__":
    X, y = get_dataset()