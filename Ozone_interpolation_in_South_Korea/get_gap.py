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
Analyze the gaps in measurement data, their length.
"""
print('analyzing existing gaps...')

print('correlated gaps...')
# read in data
y_df_flat = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_seoul/uba_graph_preproc/y.csv", index_col=0)
reg_df = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_seoul/uba_graph_preproc/reg.csv", index_col=0)
reg_df.datetime = pd.to_datetime(reg_df.datetime)
hd = HourlyData('o3')
hd.read_from_file()
y_df = hd.df
y_df.index = pd.to_datetime(y_df.index) + settings.time_offset
y_df.columns = [int(col) for col in y_df.columns]

# there are still five stations with no data, even though a
# time series existed for these. We have to make sure that
# they are not set to a value when finding the correlated
# gaps!
null_station_idx = y_df.isnull().all(axis=0)
null_station_list = y_df.columns[null_station_idx].to_list()

# set up df
columns=['station_id', 'type', 'correlated', 'start_datetime',
         'end_datetime', 'start_idx', 'len', 'n_samples']
gap_df = pd.DataFrame(columns=columns)
gap_df.index.name = 'gap_index'
gap_type = 'missing_o3_mask' ###임의의 column들로 임의의 df인 gap_df 생성

# first, find all time steps with missing data at all stations.
correlated = True
gap_idx = 0
corr_series = y_df.isnull().all(axis=1)
gap_len = 0
missing_old = corr_series[0]
datetime_gap_start = corr_series.index[0]
for datetime, missing in corr_series.iteritems():
    if missing_old == missing:
        gap_len += 1
    else:
        if missing_old:
            # found a gap. Write to all stations
            start = datetime_gap_start
            end = datetime_gap_start+pd.to_timedelta(gap_len,
                                                     unit='h')
            for station_id in y_df.columns:
                start_idx = reg_df[(reg_df.station_id==station_id) &
                            (reg_df.datetime==start)].index[0]
                # eliminate nan so the gap is not found again
                # but only if it is not part of a very long
                # gap
                if station_id not in null_station_list:
                    y_df_flat.loc[start_idx:start_idx+gap_len-1,
                                  'y'] = -999
                    # for missing o3, n_samples = gap_len
                    data = [station_id, gap_type, correlated,
                            start, end, start_idx, gap_len,
                            gap_len]
                    gap_df.loc[gap_idx, :] = data
                    gap_idx += 1
            print(f'{datetime} / {corr_series.index[-1]}') ###모든 관측소에서 데이터 결측이 발생하기 때문에 isnull() 사용
        gap_len = 1
        datetime_gap_start = datetime
    missing_old = missing

# set all values to 1 and all nans to 0
y_df_flat.loc[y_df_flat.y==y_df_flat.y, 'y'] = 1
y_df_flat.loc[y_df_flat.y!=y_df_flat.y, 'y'] = 0 ###correlated=TRUE일 경우 임의의 시간대에서 모든 관측소는 결측값이 나타남

print('single gaps...')
# find consecutive gaps and no gaps
correlated = False
y_old = y_df_flat.y[0]
station_id_old = reg_df.loc[0, 'station_id']
y_index_gap_start = y_df_flat.index[0]
gap_len = 0
for y_index, row in y_df_flat.iterrows():
    if y_index % 500000 == 0:
        print(f'{y_index/len(y_df_flat)*100:.1f} %')
    y = row.y
    station_id = reg_df.loc[y_index, 'station_id']
    if y==y_old and station_id==station_id_old:
        # increase len
        gap_len += 1
    else:
        # write gap to gap df and initialize new one
        if y_old== 0:
            gap_type = 'missing_o3_mask'
            start = reg_df.loc[y_index_gap_start, 'datetime']
            end = reg_df.loc[y_index_gap_start+gap_len, 'datetime']
            gap_df.loc[gap_idx, :] = [station_id_old, gap_type,
                                      correlated, start, end,
                                      y_index_gap_start,
                                      gap_len, gap_len] ###오존 데이터만 존재하는 y.csv 파일과 관측소 파일을 확인하며 결측값을 찾고 저장
            gap_idx += 1
        y_index_gap_start = y_index
        gap_len = 1
    y_old = y
    station_id_old = station_id ###correlated=FALSE일 경우 어떤 관측소만 결측값이 나타남

# save
gap_df.to_csv("C:/Users/ATMOS/Desktop/data/data_seoul/uba_graph_preproc/gap.csv")
