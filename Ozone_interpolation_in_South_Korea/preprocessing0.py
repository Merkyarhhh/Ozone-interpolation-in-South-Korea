import pandas as pd
import glob

folder_path = 'D:/hono/preprocessing/level3/region/Seoul/'
file_paths = glob.glob(folder_path + '*.csv')
dfs = []

for file_path in file_paths:
    df = pd.read_csv(file_path, encoding='cp949')
    df = df[(df["date"] >= "2012-01-01 00:00") & (df["date"] < "2021-12-32 00:00")]
    df = df.reset_index(drop=True)
    column_range = df.columns[df.columns.get_loc('O3'):df.columns.get_loc('RH')+1]
    for column in column_range:
        if pd.isnull(df.loc[0, column]):
            df.loc[0, column] = -999
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv('D:/기타/air_weather_merged_seoul.csv', sep=",", index=False)

columns = ["code","date","O3","NO2","SO2","CO","PM10","T","WS","WD","RH"]
df = combined_df[columns]
df.to_csv('C:/Users/ATMOS/Desktop/data_10y/country.csv', sep=",", index=False)