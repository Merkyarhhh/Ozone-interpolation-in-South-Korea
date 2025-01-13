import pandas as pd

regions = ["busan", "daegu", "daejeon", "gwangju", "incheon", "sejong", "seoul", "ulsan",
           "chungcheong", "gangwon", "gyeonggi", "gyeongsang", "jeju", "jeolla"]
pollutants = ["hod", "dow", "doy", "o3_all", "no2", "co", "so2", "pm10", "pm25", "temp", "ws", "rh"]

for region in regions:
    df1 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_hod.csv")
    df1 = pd.melt(df1, id_vars="date", value_vars=list(df1.columns[1:]), var_name="code", value_name="Hour_of_day")
    
    df2 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_dow.csv")
    df2 = pd.melt(df2, id_vars="date", value_vars=list(df2.columns[1:]), var_name="code", value_name="Day_of_week")
    df2 = df2.drop(["date", "code"], axis=1)
    df3 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_doy.csv")
    df3 = pd.melt(df3, id_vars="date", value_vars=list(df3.columns[1:]), var_name="code", value_name="Day_of_year")
    df3 = df3.drop(["date", "code"], axis=1)
    df4 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_o3_all.csv")
    df4 = pd.melt(df4, id_vars="date", value_vars=list(df4.columns[1:]), var_name="code", value_name="O3")
    df4 = df4.drop(["date", "code"], axis=1)
    df5 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_no2.csv")
    df5 = pd.melt(df5, id_vars="date", value_vars=list(df5.columns[1:]), var_name="code", value_name="NO2")
    df5 = df5.drop(["date", "code"], axis=1)
    df6 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_co.csv")
    df6 = pd.melt(df6, id_vars="date", value_vars=list(df6.columns[1:]), var_name="code", value_name="CO")
    df6 = df6.drop(["date", "code"], axis=1)
    df7 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_so2.csv")
    df7 = pd.melt(df7, id_vars="date", value_vars=list(df7.columns[1:]), var_name="code", value_name="SO2")
    df7 = df7.drop(["date", "code"], axis=1)
    df8 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_pm10.csv")
    df8 = pd.melt(df8, id_vars="date", value_vars=list(df8.columns[1:]), var_name="code", value_name="PM10")
    df8 = df8.drop(["date", "code"], axis=1)
    df9 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_pm25.csv")
    df9 = pd.melt(df9, id_vars="date", value_vars=list(df9.columns[1:]), var_name="code", value_name="PM25")
    df9 = df9.drop(["date", "code"], axis=1)
    df10 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_temp_raw.csv")
    df10 = pd.melt(df10, id_vars="date", value_vars=list(df10.columns[1:]), var_name="code", value_name="Temperature")
    df10 = df10.drop(["date", "code"], axis=1)
    df11 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_wd_raw.csv")
    df11 = pd.melt(df11, id_vars="date", value_vars=list(df11.columns[1:]), var_name="code", value_name="Wind_speed")
    df11 = df11.drop(["date", "code"], axis=1)
    df12 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_rh_raw.csv")
    df12 = pd.melt(df12, id_vars="date", value_vars=list(df12.columns[1:]), var_name="code", value_name="Relative_humidity")
    df12 = df12.drop(["date", "code"], axis=1)
    df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], axis=1)
    df.to_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/raw_{region}.csv", sep=',', index=False)
    
    

