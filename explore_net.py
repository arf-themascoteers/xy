import torch
from dataset_manager import get_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from my_machine import MyMachine

model = MyMachine()
model.load_state_dict(torch.load("model.h5"))
model.eval()

train_data, test_data = train_test_split(get_dataset(), random_state=1)

x = torch.tensor(train_data[:,0:3], dtype=torch.float32)
y = torch.tensor(train_data[:,3], dtype=torch.float32)
x.requires_grad = True
y_hat = model(x)
y_hat = y_hat.reshape(-1)
criterion = torch.nn.MSELoss(reduction='mean')
loss = criterion(y_hat, y)
loss.backward()
w = torch.abs(x.grad).sum(dim=0)
w = w/w.sum()
print(w)
