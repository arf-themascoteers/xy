import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.X = torch.rand((1000,2))
        self.y = (self.X[:,0]**2) + (self.X[:,0] * self.X[:,1])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
