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

# settings for edges, radius in km and time window
radius = 4. ###측정소 간 영향을 미칠 수 있는 거리 50km 기준 -> 한국의 특성 고려
time_window = settings.time_window ###측정소간 시차 6시간 기준 ###오존의 특성을 고려하여 임의로 정한 값

# read in data
station_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_raw/stations.csv", index_col=0)
reg_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/reg.csv", index_col=0,
                     parse_dates=['datetime'])

#reg_df = reg_df[0:int(1/10*len(reg_df))].copy() # <-- for testing

# initialize edge dataframe, n.b. it is reorganized later
edge_df = pd.DataFrame(index=reg_df.index,
                        columns=['source_indices'])
edge_df.index.name = 'target_index'

# find neighbors of stations in given radius
neighbor_df = pd.DataFrame(index=station_df.index,
                           columns=['neighbors'])
ids = station_df.index
x_, y_, z_ = geo_to_cartesian(station_df.lon.values,
                              station_df.lat.values)
coords = np.array([x_, y_, z_]).T
nn = NearestNeighbors()
nn.fit(coords)
_, idx_lists = nn.radius_neighbors(coords,radius=4.)
for trg_idx, src_idx_list in enumerate(idx_lists):
    trg_id = ids[trg_idx]
    src_ids = [ids[src_idx] for src_idx in sorted(src_idx_list)]
    neighbor_df.at[trg_id, 'neighbors'] = src_ids
    

# edges exist if measurements are at neighboring stations
# and in the given time window
def run():
    torch.multiprocessing.freeze_support()
    # edges exist if measurements are at neighboring stations
    # and in the given time window
    reg_ddf = ddf.from_pandas(reg_df, npartitions=12)
    datetime_start = datetime.datetime.now()
    print(datetime_start)
    args = (neighbor_df, reg_df, radius, time_window) ###50km 내에서 시간 차이가 6시간 이하인 경우에서 edge 선정
    edge_dseries = reg_ddf.apply(neighbor_index_filter,args=args,axis=1,meta=edge_df['source_indices'])
    edge_series = edge_dseries.compute(scheduler='multiprocessing')
    edge_df.source_indices = edge_series
    print(datetime.datetime.now() - datetime_start)
if __name__=='__main__':
    run()

# save temporary edge_df
edge_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/edge_tmp.csv")



