import pandas as pd
import numpy as np
from dataset_manager import get_dataset
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

ds = get_dataset()
ds = ds[0:100,:]
x1 = ds[:,0]
x2 = ds[:,1]
y = ds[:,2]

fig = plt.figure()
ax = plt.axes(projection='3d')

mesh = np.array(np.meshgrid(x1, x2))
Z = np.max(mesh,axis=0)
#print(Z.shape)
#exit(0)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x1, x2, Z, 50, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
print("done")


