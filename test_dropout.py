import torch
from torch import nn
from dropmachine import DropMachine

d = DropMachine()
X = torch.randn((10,3))
d(X)