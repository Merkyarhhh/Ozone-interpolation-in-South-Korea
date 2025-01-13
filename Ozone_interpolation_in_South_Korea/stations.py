import pandas as pd

df = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_all/uba_graph_raw/stations.csv")

regions = ["busan", "chungcheong", "daegu", "daejeon", "gangwon", "gwangju", "gyeonggi", "gyeongsang",
           "incheon", "jeju", "jeolla", "sejong", "seoul", "ulsan"]

for region in regions:
    df = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_all/uba_graph_raw/stations.csv")
    df1 = df[df["type_of_area"] == region]
    df1.to_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_raw/stations.csv", sep=",", index=False)