import pandas as pd

# csv 파일 불러오기
df = pd.read_csv('C:/Users/sjjun/OneDrive/Desktop/data_korea/uba_graph_raw/hourly_o3.csv')

# 같은 행의 평균값으로 결측치 채우기
df = df.fillna(df.mean(), axis=0)

df = df.rename(columns={"Unnamed: 0":"date"})


df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_korea/uba_graph_raw/hourly_cam_o3.csv", sep=",", index=False)