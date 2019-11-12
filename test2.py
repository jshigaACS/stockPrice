import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        'time':['10:00:00', '10:00:01', '10:00:02', '10:00:03', 
            '10:00:04','10:00:05', '10:00:06', '10:00:07', 
            '10:00:08', '10:00:09'],
        'val1':[25, 30, 104, 52, 41, 91, 102, 40 ,101 ,9],
        'val2':[45, 19, 34, 19, 78, 148, 45, 53 ,74 ,32]
    }
)
print('-----------')
print(df)
print('-----------')
arr = df.as_matrix()
print(arr)
print('-----------')
ravel = df.as_matrix().ravel()
print(ravel)

