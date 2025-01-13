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
import math
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
- missing_o3_mask (True if o3 is missing)
- train_mask (True if training sample)
- val_mask (True if val sample)
- test_mask (True if test sample)
n.b. all masks should have the same statistical properties as
missing_o3_mask

Also adds the new gaps to gaps.csv.
n.b. some parts of the data split are pretty hand crafted,
e.g. the masks for correlated gaps.
"""
print('preparing masks...')

# read in data
gap_df = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_busan/uba_graph_preproc/gap.csv", index_col=0)
reg_df = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_busan/uba_graph_preproc/reg.csv", index_col=0)
reg_df.datetime = pd.to_datetime(reg_df.datetime)
y_df_flat = pd.read_csv("C:/Users/ATMOS/Desktop/data/data_busan/uba_graph_preproc/y.csv", index_col=0)
hd = HourlyData('o3')
hd.read_from_file()
y_df = hd.df
y_df.index = pd.to_datetime(y_df.index) + settings.time_offset

# random seed
random.seed(settings.random_seed)

# initialize mask df
columns = ['missing_o3_mask', 'train_mask', 'val_mask',
           'test_mask']
mask_df = pd.DataFrame(columns=columns, index=y_df_flat.index,
                       data=False)
mask_df.index.name = 'node_index' ###missing_o3, train, val, test_mask를 column으로 임의의 df 생성

# reset gap data frame
gap_df = gap_df[gap_df.type=='missing_o3_mask'].copy()

# missing o3 is simply where we do not have any measurement
mask_df.loc[y_df_flat.y!=y_df_flat.y, 'missing_o3_mask'] = True ### [!=] 결측치를 True로 설정

# shopping list for single gaps
shopping_list = []
n_samples_required = 0
# Calculate the number of samples
n_samples = len(mask_df)

# Calculate the count of False values in the 'missing_o3_mask' column
false_count = mask_df[~mask_df['missing_o3_mask']].shape[0]

# Get the value from the 'missing_o3_mask' column where it's True
true_count = mask_df[mask_df['missing_o3_mask']]['missing_o3_mask'].shape[0]

# Calculate the n_ratio
n_ratio = round(false_count * 0.15)

# Calculate the ratio
ratio = n_ratio / float(true_count)

print('\nshopping list each for val and test set')
for idx, len_ in enumerate(lens):
    n_gaps_required = round(counts[idx] * ratio)
    if n_gaps_required < 1:
        continue
    # split year long gaps in half
    if len_ == 8760:
        len_ = round(len_ / 2)
        n_gaps_required *= 2
    n_samples = n_gaps_required * len_
    n_samples_required += n_samples
    
    print(f'length {len_}\t# gaps {n_gaps_required}\t# ' +
          f'samples {n_samples}')
    shopping_list.append((len_, n_gaps_required))
n_samples_required *= 2  # bc we have test and val sets
print(f'\t\tsum: \t{n_samples_required} samples = ' +
      f'{n_samples_required/n_total*100:.1f} %')


# train mask is where no other mask applies
index_list = mask_df[(mask_df==False).all(axis=1)].index
mask_df.loc[index_list, 'train_mask'] = True ###만약 다른 항목들이 모두 Flase라면 train_mask를 True로 입력

# convert gap df columns back to int
gap_df = gap_df.astype({'station_id':int, 'start_idx':int,
                        'len':int, 'n_samples':int})

# print a summary of all masks
print('\nmask summary')
n_samples = len(mask_df)
print(f'{n_samples} in total (100.0 %)')
for col in mask_df.columns:
    n_samples_set = mask_df[col].sum().sum()
    percentage = n_samples_set / n_samples * 100
    print(f'{n_samples_set} in {col} ({percentage:.2f} %)')

# mask sanity check
assert mask_df.sum().sum() == len(mask_df)

# save
mask_df.to_csv("C:/Users/ATMOS/Desktop/data/data_busan/uba_graph_preproc/mask.csv")

