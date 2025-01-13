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
A file with all the labels (i.e. ozone values)
"""
print('preprocessing y...')
hd = HourlyData('temp')
hd.read_from_file()

# time offset
time_offset = settings.time_offset
hd.df.index = pd.to_datetime(hd.df.index) + time_offset

# flatten, but move through time steps first.
y_values = hd.df.values.T.reshape(-1)
y_df = pd.DataFrame(index=range(len(y_values)),
                    columns=['y'],
                    data=y_values) ###raw data-hourlyO3에서 첫 번째 열의 값만 추출하여 df 생성
y_df.index.name = 'node_index' ###node_index 부여

# save
y_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/y.csv")
