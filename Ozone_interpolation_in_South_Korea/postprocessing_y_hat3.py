import pandas as pd
import numpy as np

regions = ["busan", "daegu", "daejeon", "gwangju", "incheon", "sejong", "seoul", "ulsan",
           "chungcheong", "gangwon", "gyeonggi", "gyeongsang", "jeolla", "jeju"]

for region in regions:
    columns = ['date1']
    df0 = pd.DataFrame(columns=columns)
    date_range = pd.date_range(start="2021-01-01 00:00:00", end="2021-12-31 23:00:00", freq="H")
    df0 = pd.DataFrame({'date1': date_range})
    df = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/output/characteristics_tmp.csv", encoding="cp949")
    
    
    columns = ['station_id', 'date', 'month', 'hour','y_true', 'y_imputed', 'O3', 'O3_abs', 'date1', 'test_mask']
    df1 = pd.DataFrame(columns=columns)
    df1["station_id"] = df["station_id"]
    df1["date"] = df["node_index"]
    df1["date1"] = df0["date1"]
    df1["y_true"] = df["y_true"]
    df1["y_imputed"] = df["y_imputed"]
    df1["O3"] = (df["y_imputed"] - df["y_true"])
    df1["O3_abs"] = abs(df["y_imputed"] - df["y_true"])
    df1['test_mask'] = df['test_mask']
    
    size = len(df1["date"]) // 8760
    date1_set = df1['date1'].head(8760)
    date1_set_repeated = pd.concat([date1_set] * size, ignore_index=True)
    df1['date'] = date1_set_repeated
    
    df1 = df1.drop(["date1"], axis=1)
    df1["date"] = pd.to_datetime(df1["date"])
    df1["month"] = df1["date"].dt.month
    df1["hour"] = df1["date"].dt.hour
    
    df1.loc[df1["month"].isin([1]), "month"] = "2021-01"
    df1.loc[df1["month"].isin([2]), "month"] = "2021-02"
    df1.loc[df1["month"].isin([3]), "month"] = "2021-03"
    df1.loc[df1["month"].isin([4]), "month"] = "2021-04"
    df1.loc[df1["month"].isin([5]), "month"] = "2021-05"
    df1.loc[df1["month"].isin([6]), "month"] = "2021-06"
    df1.loc[df1["month"].isin([7]), "month"] = "2021-07"
    df1.loc[df1["month"].isin([8]), "month"] = "2021-08"
    df1.loc[df1["month"].isin([9]), "month"] = "2021-09"
    df1.loc[df1["month"].isin([10]), "month"] = "2021-10"
    df1.loc[df1["month"].isin([11]), "month"] = "2021-11"
    df1.loc[df1["month"].isin([12]), "month"] = "2021-12"
    
    df1 = df1[df1["test_mask"] == True]
    
    df1.to_csv(f"C:/Users/ATMOS/Desktop/data_fin/y_hat/y_hat_{region}.csv", sep=",", index=False)
    
dfs =[]

for region in regions:
    df = pd.read_csv(f"C:/Users/ATMOS/Desktop/data_fin/y_hat/y_hat_{region}.csv", encoding="cp949")
    dfs.append(df)

concatenated_df1 = pd.concat(dfs, ignore_index=False)
concatenated_df1.to_csv("C:/Users/ATMOS/Desktop/data_fin/y_hat/y_hat.csv", sep=",", index=False)
concatenated_df2 = concatenated_df1.groupby("date").mean()
concatenated_df2.to_csv("C:/Users/ATMOS/Desktop/data_fin/y_hat/y_hat_average.csv", sep=",", index=False)

