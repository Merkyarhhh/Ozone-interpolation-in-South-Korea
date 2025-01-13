import pandas as pd

regions = ["busan", "daegu", "daejeon", "gwangju", "incheon", "sejong", "seoul", "ulsan"]
           #"chungcheong", "gangwon", "gyeonggi", "gyeongsang", "jeju", "jeolla"]

for region in regions:
    edge_df = pd.read_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_preproc/edge_tmp.csv", index_col=0)
    source_index_list, target_index_list = [], []
    for index, row in edge_df.iterrows():
        source_indices = row.source_indices.strip('][').split(', ')
        source_indices = [int(idx) for idx in source_indices]
        target_index = index
        source_index_list += source_indices
        target_index_list += [target_index]*len(source_indices)
    edge_df = pd.DataFrame()
    edge_df['source_node_index'] = source_index_list
    edge_df['target_node_index'] = target_index_list
    edge_df.index.name = 'edge_index'
    edge_df.to_csv(f"C:/Users/ATMOS/Desktop/data/data_{region}/uba_graph_preproc/edge.csv")


