from my_machine import MyMachine
from my_dataset import MyDataset
from torch.utils.data import DataLoader
import torch


def train():
    NUM_EPOCHS = 800
    BATCH_SIZE = 2000

    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = MyMachine()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.MSELoss(reduction='mean')

    for epoch in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)
            y_pred = y_pred.reshape(-1)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    print("Training done. Machine saved to models/model.h5")
    torch.save(model.state_dict(), 'model.h5')
    return model


if __name__ == "__main__":
    train()
