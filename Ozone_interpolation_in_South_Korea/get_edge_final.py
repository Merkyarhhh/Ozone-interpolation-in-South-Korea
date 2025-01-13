import datetime
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


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

def loc_weight(loc_target_node, loc_source_nodes):
    """
    For two data frames, containing cartesian coordinates, return
    weights by distance (0km => weight=1 ; 50km => weight=0)
    """
    radius = 4.
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
    time_window = pd.Timedelta('0 days 06:00:00').components.hours
    differences = (time_source_nodes - time_target_node.iloc[0]).astype(
                                               'timedelta64[h]').values
    assert np.all((differences <= time_window))

    weights = (time_window - differences) / time_window

    return weights
radius = 4.
time_window = pd.Timedelta('0 days 06:00:00')



station_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_raw/stations.csv", index_col=0)
reg_df = pd.read_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_graph_preproc/reg.csv", index_col=0,
                     parse_dates=['datetime'])
edge_df = pd.DataFrame(index=reg_df.index,
                        columns=['source_indices'])
edge_df.index.name = 'target_index'
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


def run(name):
    
    args = (neighbor_df, reg_df, radius, time_window)
    edge_series = reg_df.apply(neighbor_index_filter, args=args, axis=1)
    edge_df['source_indices'] = edge_series
    edge_df.to_csv("C:/Users/sjjun/OneDrive/Desktop/data_seoul/uba_preproc/edge_tmp.csv")
if __name__ == '__main__':
    datetime_start = datetime.datetime.now()
    print(datetime_start)
    procs = []
    for i in range(12):
        p = multiprocessing.Process(target=run, args=(i, ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()    
    print(datetime.datetime.now() - datetime_start)





