# general
import pdb

# data science
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances

# own package
import settings


def geo_to_cartesian(lons, lats):
    """
    Maps longitudes and latitudes onto a 3d grid which is practical
    for distance measures

    inputs: lons, lats in degree, dtype is series.
    returns: numpy arrays, in km
    """
    r = 6371.

    lons = lons * np.pi / 180.
    lats = lats * np.pi / 180.

    x = r * np.cos(lats) * np.cos(lons)
    y = r * np.cos(lats) * np.sin(lons)
    z = r * np.sin(lats)

    return x, y, z


def neighbor_index_filter(row, neighbor_df, reg_df, radius, time_window):
    """
    For a given row in reg_df, find all neighboring nodes
    """

    trg_id = row.station_id
    trg_datetime = row.datetime
    neigh_ids = neighbor_df.at[trg_id, 'neighbors']

    rad_filter = reg_df.station_id.isin(neigh_ids)
    time_filter = abs(reg_df.datetime-trg_datetime) <= time_window

    src_idxs = reg_df[rad_filter&time_filter].index.to_list()

    return src_idxs


def get_one_hot(x_df):
    """
    If this data frame contains 'type' or 'type_of_area',
    replace them with one-hot encoded columns
    """

    for col in ['type', 'type_of_area']:
        if col in x_df.columns:
            dummies = pd.get_dummies(x_df[col], prefix=col)
            x_df.drop(col, axis=1, inplace=True)
            x_df = pd.concat([x_df, dummies], axis=1)

    return x_df.copy()


def loc_weight(loc_target_node, loc_source_nodes):
    """
    For two data frames, containing cartesian coordinates, return
    weights by distance (0km => weight=1 ; 50km => weight=0)
    """
    radius = settings.radius
    distances = pairwise_distances(loc_target_node,
                                   loc_source_nodes).reshape(-1)
    assert np.all((distances <= radius))

    weights = (radius - distances) / radius

    return weights


def time_weight(time_target_node, time_source_nodes):
    """
    For two data frames, containing datetimes, return
    weights by time offsets (0h => weight=1 ; 6h => weight=0)
    """
    time_window = settings.time_window.components.hours
    differences = (time_source_nodes - time_target_node.iloc[0]).astype(
                                               'timedelta64[h]').values
    assert np.all((differences <= time_window))

    weights = (time_window - differences) / time_window

    return weights

