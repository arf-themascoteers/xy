import torch


def get_dataset():
        X = torch.rand((1000,2))
        y = (X[:,0]**2) + (X[:,0] * X[:,1])
        return X, y

