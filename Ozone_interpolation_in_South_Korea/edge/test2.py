import datetime
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

edge_tmp_df = pd.read_csv("/home/tech/tech1/atmos_KU/edge_tmp.csv", index_col=0)
pos_df = pd.read_csv("/home/tech/tech1/atmos_KU/pos.csv", index_col=0, converters={'datetime': pd.Timestamp})
loc_df = pd.DataFrame()
loc_df['x'], loc_df['y'], loc_df['z'] = geo_to_cartesian(
                                        pos_df.lon, pos_df.lat) 
edge_weight_list = []
for target_node_index in edge_tmp_df.index:
    if target_node_index % 50000 == 0:
        print(f'{target_node_index/len(edge_tmp_df)*100:.2f} %')
    source_node_indices = eval(edge_tmp_df.at[target_node_index,
                                              'source_indices'])
    loc_target_node = loc_df.loc[[target_node_index]]
    loc_source_nodes = loc_df.loc[source_node_indices]
    time_target_node = pos_df.loc[[target_node_index], 'datetime']
    time_source_nodes = pos_df.loc[source_node_indices, 'datetime']
    weights_source_nodes = loc_weight(loc_target_node, loc_source_nodes) * \
                           time_weight(time_target_node, time_source_nodes)
    weights_source_nodes = weights_source_nodes/sum(weights_source_nodes)
    edge_weight_list += weights_source_nodes.tolist()

weight_df = pd.DataFrame(columns=['edge_weight'])
weight_df['edge_weight'] = edge_weight_list
weight_df.index.name = 'edge_index'
weight_df.to_csv("/home/tech/tech1/atmos_KU/weight.csv")



edge_df = pd.read_csv("/home/tech/tech1/atmos_KU/edge_tmp.csv", index_col=0)
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
edge_df.to_csv("/home/tech/tech1/atmos_KU/edge.csv")
