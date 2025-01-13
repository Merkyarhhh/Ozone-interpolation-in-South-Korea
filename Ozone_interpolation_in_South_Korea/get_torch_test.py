    """
Creates the pytorch geometric dataset that is used in our publication.
To reuse the dataset, use the UBAGraph.get_dataset() routine.
"""

# general

import warnings


# data science
import numpy as np
import pandas as pd


# pytorch
import torch
from preprocessing_utils import geo_to_cartesian, neighbor_index_filter, \
                                loc_weight, time_weight, get_one_hot


# oppress future warnings
warnings.filterwarnings('ignore')

"""
Convert all the .csv files to tensors
"""
print('converting to pytorch tensors...')

# x
x_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/x.csv", index_col=0)
x_df = get_one_hot(x_df)
x = torch.tensor(x_df.values.astype(np.float32))
torch.save(x, "C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/x.pt")
print('C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/x.pt')
del x_df

# y
y_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/y.csv", index_col=0)
y = torch.tensor(y_df.values.astype(np.float32)).view(-1, 1)
torch.save(y, "C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/y.pt")
print("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/y.pt")
del y_df

# masks
mask_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/mask.csv", index_col=0)
missing_o3_mask = torch.tensor(mask_df.missing_o3_mask).view(-1)
torch.save(missing_o3_mask, 'C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/missing_o3_mask.pt')
print('C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/missing_o3_mask.pt')
train_mask = torch.tensor(mask_df.train_mask).view(-1)
torch.save(train_mask, 'C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/train_mask.pt')
print('C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/train_mask.pt')
val_mask = torch.tensor(mask_df.val_mask).view(-1)
torch.save(val_mask, 'C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/val_mask.pt')
print('C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/val_mask.pt')
test_mask = torch.tensor(mask_df.test_mask).view(-1)
torch.save(test_mask, 'C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/test_mask.pt')
print('C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/test_mask.pt')
del mask_df

# position
pos_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/pos.csv", index_col=0)
pos = torch.tensor(pos_df[['lon', 'lat']].values.astype(np.float32))
torch.save(pos, 'C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/pos.pt')
print('C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_dataset/pos.pt')
del pos_df