import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_gyeonggi/uba_graph_raw/hourly_o3.csv", encoding='cp949')
df0 = df['date']
np.random.seed(1)
df1 = df['131222.0'].sample(frac=0.95, replace=False)
df2 = df['131222.0'].sample(frac=0.85, replace=False)
df3 = df['131222.0'].sample(frac=0.75, replace=False)
df4 = df['131222.0'].sample(frac=0.65, replace=False)
df5 = df['131222.0'].sample(frac=0.55, replace=False)
df6 = df['131222.0'].sample(frac=0.45, replace=False)
df7 = df['131222.0'].sample(frac=0.35, replace=False)
df8 = df['131222.0'].sample(frac=0.25, replace=False)



dfs = [df0,df1,df2,df3,df4,df5,df6,df7,df8]

df0 = pd.concat(dfs, axis=1)

df0.to_csv('C:/Users/ATMOS/Desktop/random_test/gyeonggi/random_sampling.csv')