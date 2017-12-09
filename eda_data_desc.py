import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



then = time.time()

train = pd.read_csv("./train.csv")

now = time.time()
print(now - then)
then = now

test = pd.read_csv("./test.csv")

now = time.time()
print(now - then)

print(train.head(6))
print(train.describe())

print(test.head(6))
print(test.describe())