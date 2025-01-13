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

"""
Position of every node: lon, lat, datetime
"""
print('preprocessing positions of nodes...')

# read in data
reg_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/reg.csv", index_col=0)
sd = StationData()
sd.read_from_file()
station_df = sd.df ###raw data-stations를 불러오기

# initialize position data frame
pos_df = pd.DataFrame(index=reg_df.index,
                      columns=['lon', 'lat', 'datetime'])
pos_df.index.name = 'node_index' ###필요한 column만을 선별하여 pos 데이터프레임 생성

# write positions to df
for idx, col in station_df.iterrows():
    id_filter = reg_df.station_id==idx
    for coordinate in ['lon', 'lat']:
        pos_df.loc[id_filter, coordinate] = col[coordinate]
pos_df.datetime = reg_df.datetime ###필요한 column인 위도와 경도를 stations 데이터에서 불러와 pos 데이터프레임에 입력

# save
pos_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/pos.csv")
