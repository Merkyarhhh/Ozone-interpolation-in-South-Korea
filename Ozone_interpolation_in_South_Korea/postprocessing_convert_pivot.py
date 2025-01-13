import os
import pandas as pd

regions = ["busan", "daegu", "daejeon", "gwangju", "incheon", "sejong", "seoul", "ulsan",
           "chungcheong", "gangwon", "gyeonggi", "gyeongsang", "jeju", "jeolla"]
    #"busan", "daegu", "daejeon", "gwangju", "incheon", "sejong", "seoul", "ulsan"]
           #"chungcheong", "gangwon", "gyeonggi", "gyeongsang", "jeju", "jeolla"]

for region in regions:
    df1 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/hourly_o3.csv")
    df2 = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/imputed_dataset/imputed_o3.csv")

    df1 = pd.melt(df1, id_vars="date", value_vars=list(df1.columns[1:]), var_name="code", value_name="O3")
    df2 = pd.melt(df2, id_vars="date", value_vars=list(df2.columns[1:]), var_name="code", value_name="O3")

    #df1.to_csv(f"C:/Users/ATMOS/Desktop/data_fin/hourly_o3_{region}.csv", sep=',', index=False)
    #df2.to_csv(f"C:/Users/ATMOS/Desktop/data_fin/imputed_o3_{region}.csv", sep=',', index=False)

    df1.insert(1, column="type", value="raw")
    df2.insert(1, column="type", value="imputed")

    df = pd.concat([df1,df2])
    df.to_csv(f"C:/Users/ATMOS/Desktop/data_fin/total_o3_{region}.csv", sep=',', index=False)
    
    

merged_df = pd.DataFrame()
base_path = "C:/Users/ATMOS/Desktop/data/"

for region in regions:
    o3_csv_path = os.path.join(base_path, f"data_{region}/uba_graph_raw/hourly_o3.csv")
    imputed_csv_path = os.path.join(base_path, f"data_{region}/imputed_dataset/imputed_o3.csv")
    df1 = pd.read_csv(o3_csv_path)
    df2 = pd.read_csv(imputed_csv_path)
    
    df1 = pd.melt(df1, id_vars="date", value_vars=list(df1.columns[1:]), var_name="code", value_name="O3")
    df2 = pd.melt(df2, id_vars="date", value_vars=list(df2.columns[1:]), var_name="code", value_name="O3")
    df1 = df1.rename(columns={"O3":"O3_raw"})
    df2 = df2.rename(columns={"O3":"O3_imputed"})
    df = pd.merge(df1,df2, on = ["date", "code"], how = "outer")
    merged_df = pd.concat([merged_df, df])
    merged_df.to_csv("C:/Users/ATMOS/Desktop/data_fin/total_o3.csv", sep=',', index=False)
    
    

df = pd.read_csv("C:/Users/ATMOS/Desktop/data_fin/total_o3.csv")
df = df.groupby('code').mean()
df.to_csv("C:/Users/ATMOS/Desktop/data_fin/total_o3_yearly.csv", sep=',', index=False)


o3_presence_by_code = df.groupby('code')['O3_raw'].apply(lambda x: (x.notnull().sum() / len(x)) * 100).reset_index()
o3_presence_by_code.rename(columns={'O3_raw': 'O3_presence_percentage'}, inplace=True)

o3_presence_by_code.to_csv("C:/Users/ATMOS/Desktop/data_fin/recovery.csv", index=False)