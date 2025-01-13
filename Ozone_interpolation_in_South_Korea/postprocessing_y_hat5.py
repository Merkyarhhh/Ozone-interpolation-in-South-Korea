import pandas as pd
import numpy as np

regions = ["busan", "daegu", "daejeon", "gwangju", "incheon", "sejong", "seoul", "ulsan",
           "chungcheong", "gangwon", "gyeonggi", "gyeongsang", "jeolla", "jeju"]

for region in regions:
    df = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/output/characteristics_tmp.csv", encoding="cp949")
    
    df = df[df["test_mask"] == True]
    
    df.to_csv(f"C:/Users/ATMOS/Desktop/data_fin/y_hat0/y_hat0_{region}.csv", sep=",", index=False)
    
dfs =[]

for region in regions:
    df = pd.read_csv(f"C:/Users/ATMOS/Desktop/data_fin/y_hat0/y_hat0_{region}.csv", encoding="cp949")
    dfs.append(df)

concatenated_df1 = pd.concat(dfs, ignore_index=False)
concatenated_df1.to_csv("C:/Users/ATMOS/Desktop/data_fin/y_hat0/y_hat0.csv", sep=",", index=False)
