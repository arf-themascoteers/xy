import pandas as pd
import numpy as np


def make():
    x = np.random.uniform(0, 1, (1000,3))
    x[:,2] = (x[:,0]**2) * x[:,1] + np.log(x[:,1]) + (x[:,1]**2) #(x[:,0]**2)  + (x[:,1]**2)

    columns = ["x1", "x2", "y"]
    df = pd.DataFrame(data=x, columns=columns)
    df.to_csv("data.csv",index=False)


if __name__ == "__main__":
    make()
