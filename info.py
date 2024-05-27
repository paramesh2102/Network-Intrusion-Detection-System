import numpy
print('hello world')
import platform
print(platform.python_version())
import pandas as pd
import numpy as np

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

a=pd.DataFrame({'a':pd.Series([1,2,3]),'b':pd.Series(['a','b','c'])})
print(pd.get_dummies((a)))


data=pd.read_csv('/nids_website/app/maxdf.csv')
print(data.head(2))
