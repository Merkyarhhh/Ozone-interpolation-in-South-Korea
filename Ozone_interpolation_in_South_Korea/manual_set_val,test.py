import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/val.csv")
df = df.fillna(value=-1, inplace=False)
df = df.replace(to_replace=df[df!=-1].values, value=300)
df.to_csv("C:/Users/sjjun/OneDrive/Desktop/val.csv")

df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/test.csv")
df = df.fillna(value=-1, inplace=False)
df = df.replace(to_replace=df[df!=-1].values, value=301)
df.to_csv("C:/Users/sjjun/OneDrive/Desktop/test.csv")
