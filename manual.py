import pandas as pd
import torch
import numpy as np
import dataset_manager
from sklearn.metrics import r2_score

ds = dataset_manager.get_dataset()
X = ds[:,0:-1]
y = ds[:,-1]
a = X[:,0]* (2.1928e-07) + X[:,1]* (1.4416e+00) + (2.7894e-01)
b = X[:,0]* (1.2105e+00) + X[:,1]* (-1.2105e+00) + (1.8322e-06)
c = X[:,0]* (-6.9637e-06) + X[:,1]* (-1.2918e+00) + (-2.4995e-01)

a[a<0] = 0
b[b<0] = 0
c[c<0] = 0

d = a * (0.6937,) + b * (0.8261) + c* (-0.7741) + (-0.1935)

print(r2_score(y,d))