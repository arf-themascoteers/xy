import torch
import numpy as np
from my_machine import MyMachine
import dataset_manager
from sklearn.metrics import r2_score

ds = dataset_manager.get_dataset()
X = ds[:,0:-1]

model = MyMachine(X.shape[1])
model.load_state_dict(torch.load("model.h5"))
model.eval()

for p in model.parameters():
    print(p)