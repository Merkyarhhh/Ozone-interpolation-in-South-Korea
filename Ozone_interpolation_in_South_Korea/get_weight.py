"""
Creates the pytorch geometric dataset that is used in our publication.
To reuse the dataset, use the UBAGraph.get_dataset() routine.
"""

# general
import random
import pdb
import warnings
import pickle as pkl
import datetime

# data science
import numpy as np
import pandas as pd
import dask.dataframe as ddf

# pytorch
import torch
from torch_geometric.data import Data, InMemoryDataset

# sklearn
from sklearn.neighbors import NearestNeighbors

# own package
import settings
from retrieval_toar_db import StationData
from retrieval_toar_db import HourlyData
from preprocessing_utils import geo_to_cartesian, neighbor_index_filter, \
                                loc_weight, time_weight, get_one_hot


# oppress future warnings
warnings.filterwarnings('ignore')

"""
Weight edges by spatial and temporal proximity
"""
print('preprocessing edge weights...')

# read in data
edge_tmp_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/edge_tmp.csv", index_col=0)
pos_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/pos.csv", index_col=0,
                     converters={'datetime': pd.Timestamp})

# locations
loc_df = pd.DataFrame()
loc_df['x'], loc_df['y'], loc_df['z'] = geo_to_cartesian(
                                        pos_df.lon, pos_df.lat) ###좌표계 설정

# iterate through all node indices
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
                           time_weight(time_target_node, time_source_nodes) ###기존의 target_node와 source_node를 시간과 공간을 기준으로 곱함
    weights_source_nodes = weights_source_nodes/sum(weights_source_nodes) ###이 값들을 합계값으로 나눠서 표현
    edge_weight_list += weights_source_nodes.tolist()

# edge weight dataframe
weight_df = pd.DataFrame(columns=['edge_weight'])
weight_df['edge_weight'] = edge_weight_list ###계산한 edge_weight_list를 edege_weight column에 대입
weight_df.index.name = 'edge_index'

# save
weight_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/weight.csv")
