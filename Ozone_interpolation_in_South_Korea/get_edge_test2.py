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
import multiprocessing

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
Collect all edges. This routine best runs on the mem192
partition of JUWELS.

This routine also produces the edge_tmp.csv file,
which makes calculating edge weights a lot faster.
"""
print('preprocessing edges...')

# settings for edges, radius in km and time window
radius = settings.radius ###측정소 간 영향을 미칠 수 있는 거리 50km 기준 -> 한국의 특성 고려
time_window = settings.time_window ###측정소간 시차 6시간 기준 ###오존의 특성을 고려하여 임의로 정한 값

# read in data
edge_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/edge_tmp.csv", index_col=0)



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
edge_df.index.name = 'edge_index' ###source_list = ([1,2,3], [1,2,3,4], ...) -> (1,2,3,1,2,3,4, ...) 전개

# save
edge_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/edge.csv")

