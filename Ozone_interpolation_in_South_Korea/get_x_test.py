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
A file with all node features
"""
print('preprocessing x...')

# organize features
easy_names = ['hour_of_day', 'day_of_week', 'day_of_year']
rea_names = ['temp', 'ws']
cams_names = ['cams_co', 'cams_so2', 'cams_no2', 'cams_pm10']
meta_names = ['type', 'type_of_area'] ###raw data에서 위의 항목을 모두 불러오기

# set up df
feature_names = easy_names + rea_names + cams_names + \
                meta_names ###불러온 항목을 모두 합치기
reg_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/reg.csv", index_col=0,
                     converters={'datetime': pd.Timestamp}) ###datetime값을 변환
x_df = pd.DataFrame(columns=feature_names, index=reg_df.index) ###reg데이터에서 만든 node_index를 부여, 불러온 항목으로 df 생성

# easy fields
x_df.hour_of_day = [d.hour for d in reg_df.datetime]
x_df.day_of_week = [d.dayofweek for d in reg_df.datetime]
x_df.day_of_year = [d.dayofyear for d in reg_df.datetime] ###datetime를 다음 항목으로 변환

# rea fields, and cams fields, add time offset
for name in rea_names+cams_names:
    hd = HourlyData(name)
    hd.read_from_file()
    time_offset = settings.time_offset
    hd.df.index = pd.to_datetime(hd.df.index) + time_offset
    x_df[name] = hd.df.values.T.reshape(-1) ###시간단위로 측정된 데이터에 time_offset를 부여하고 첫 번째 열의 값만 추출

# meta fields
sd = StationData()
sd.read_from_file()
for idx, col in sd.df.iterrows():
    id_filter = reg_df.station_id==idx
    for meta_name in meta_names:
        x_df.loc[id_filter, meta_name] = col[meta_name] ###station의 고유한 데이터를 df에 필요한 값만 부여

# remame fields to wish names
mapper = {'temp': 'temperature',
          'ws': 'wind_speed'}
x_df.rename(columns=mapper, inplace=True) ###column명 전환

# save
x_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/x.csv")
