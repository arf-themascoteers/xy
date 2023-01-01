import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, is_train=True):
        torch.manual_seed(0)
        X = torch.rand((1000,2))
        y = (X[:,0]**2) + (X[:,0] * X[:,1])

        self.X, X_test, self.y, y_test = train_test_split(X, y, random_state=1, test_size=0.4)

        if not is_train:
            self.X = X_test
            self.y = y_test

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]