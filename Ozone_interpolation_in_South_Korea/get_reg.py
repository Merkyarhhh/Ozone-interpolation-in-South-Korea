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
A register file, where the sample id, the corresponding station
id and the datetime can be read.
"""
print('preprocessing register...')

# example data
hd = HourlyData('o3')
hd.read_from_file() ###raw data-hourlyO3를 불러오기

# filter the datetime index and add time offset
time_offset = settings.time_offset
hd.df.index = pd.to_datetime(hd.df.index) + time_offset ###hourlyO3에서 시간에 time_offset인 2시간 추가

# set up df
n_samples_total = len(hd.df.index)*len(hd.df.columns) ###총 데이터 셀을 계산
cols = ['station_id', 'datetime']
reg_df = pd.DataFrame(index=range(n_samples_total),
                      columns=cols) ### 
reg_df.index.name = 'node_index' ###필요한 column만을 선별

# fill df
idx = 0
for station_idx in hd.df.columns:
    for time_idx in hd.df.index:
        reg_df.loc[idx] = [station_idx, time_idx]
        idx += 1
        if idx % 500000 == 0:
            print(f'{idx/len(reg_df)*100:.1f} %') ###필요한 column인 id를 stations 데이터에서 불러와 reg 데이터프레임에 입력

# save
reg_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/reg.csv")