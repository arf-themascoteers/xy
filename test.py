import torch
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from my_machine import MyMachine
from sklearn.metrics import r2_score


def test():
    BATCH_SIZE = 2000
    dataset = MyDataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = MyMachine()
    model.load_state_dict(torch.load("model.h5"))
    model.eval()
    print(f"Test started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            print(r2_score(y_true, y_pred))


if __name__ == "__main__":
    test()
