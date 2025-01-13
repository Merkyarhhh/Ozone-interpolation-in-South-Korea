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


class UBAGraph(InMemoryDataset):
    """
    A dataset for graph ML on time resolved
    ozone data.
    """
    def __init__(self):
        """
        Initialize the class
        """
        # super class
        InMemoryDataset.__init__(self)

        # raw data paths
        self.raw_data_dir = settings.resources_dir + 'uba_graph_raw/'
        self.station_path = self.raw_data_dir + 'stations.csv'

        # preprocessed data paths
        self.out_dir = settings.resources_dir + 'uba_graph_preproc/'
        self.reg_path = self.out_dir + 'reg.csv'
        self.pos_path = self.out_dir + 'pos.csv'
        self.x_path = self.out_dir + 'x.csv'
        self.y_path = self.out_dir + 'y.csv'
        self.gap_path = self.out_dir + 'gap.csv'
        self.mask_path = self.out_dir + 'mask.csv'
        self.edge_path = self.out_dir + 'edge.csv'
        self.edge_tmp_path = self.out_dir + 'edge_tmp.csv'
        self.weight_path = self.out_dir + 'weight.csv'

        # torch dataset paths
        self.dataset_dir = settings.resources_dir + 'uba_graph_dataset/'
        self.x_pt_path = self.dataset_dir + 'x.pt'
        self.edge_index_pt_path = self.dataset_dir + 'edge_index.pt'
        self.edge_weight_pt_path = self.dataset_dir + 'edge_weights.pt'
        self.y_pt_path = self.dataset_dir + 'y.pt'
        self.missing_o3_mask_pt_path = self.dataset_dir + 'missing_o3_mask.pt'
        self.train_mask_pt_path = self.dataset_dir + 'train_mask.pt'
        self.val_mask_pt_path = self.dataset_dir + 'val_mask.pt'
        self.test_mask_pt_path = self.dataset_dir + 'test_mask.pt'
        self.pos_pt_path = self.dataset_dir + 'pos.pt'

    def get_reg(self):
        """
        A register file, where the sample id, the corresponding station
        id and the datetime can be read.
        """
        print('preprocessing register...')

        # example data
        hd = HourlyData('o3')
        hd.read_from_file()

        # filter the datetime index and add time offset
        time_offset = settings.time_offset
        hd.df.index = pd.to_datetime(hd.df.index) + time_offset

        # set up df
        n_samples_total = len(hd.df.index)*len(hd.df.columns)
        cols = ['station_id', 'datetime']
        reg_df = pd.DataFrame(index=range(n_samples_total),
                              columns=cols)
        reg_df.index.name = 'node_index'

        # fill df
        idx = 0
        for station_idx in hd.df.columns:
            for time_idx in hd.df.index:
                reg_df.loc[idx] = [station_idx, time_idx]
                idx += 1
                if idx % 500000 == 0:
                    print(f'{idx/len(reg_df)*100:.1f} %')

        # save
        reg_df.to_csv(self.reg_path)
        print(f'written to {self.reg_path}')

    def get_pos(self):
        """
        Position of every node: lon, lat, datetime
        """
        print('preprocessing positions of nodes...')

        # read in data
        reg_df = pd.read_csv(self.reg_path, index_col=0)
        sd = StationData()
        sd.read_from_file()
        station_df = sd.df

        # initialize position data frame
        pos_df = pd.DataFrame(index=reg_df.index,
                              columns=['lon', 'lat', 'datetime'])
        pos_df.index.name = 'node_index'

        # write positions to df
        for idx, col in station_df.iterrows():
            id_filter = reg_df.station_id==idx
            for coordinate in ['lon', 'lat']:
                pos_df.loc[id_filter, coordinate] = col[coordinate]
        pos_df.datetime = reg_df.datetime

        # save
        pos_df.to_csv(self.pos_path)
        print(f'written to {self.pos_path}')

    def get_x(self):
        """
        A file with all node features
        """
        print('preprocessing x...')

        # organize features
        easy_names = ['hour_of_day', 'day_of_week', 'day_of_year']
        rea_names = ['temp', 'ws', 'rh']
        cams_names = ['cams_no2', 'cams_co', 'cams_so2',
                      'cams_pm10', 'cams_pm25']

        # set up df
        feature_names = easy_names + rea_names + cams_names
        reg_df = pd.read_csv(self.reg_path, index_col=0,
                             converters={'datetime': pd.Timestamp})
        x_df = pd.DataFrame(columns=feature_names, index=reg_df.index)

        # easy fields
        x_df.hour_of_day = [d.hour for d in reg_df.datetime]
        x_df.day_of_week = [d.dayofweek for d in reg_df.datetime]
        x_df.day_of_year = [d.dayofyear for d in reg_df.datetime]

        # rea fields, and cams fields, add time offset
        for name in rea_names+cams_names:
            hd = HourlyData(name)
            hd.read_from_file()
            time_offset = settings.time_offset
            hd.df.index = pd.to_datetime(hd.df.index) + time_offset
            x_df[name] = hd.df.values.T.reshape(-1)

        # remame fields to wish names
        mapper = {'temp': 'temperature',
                 'ws': 'wind_speed',
                 'rh': 'relative_humidity',}
        x_df.rename(columns=mapper, inplace=True)

        # save
        x_df.to_csv(self.x_path)
        print(f'written to {self.x_path}')

    def get_y(self):
        """
        A file with all the labels (i.e. ozone values)
        """
        print('preprocessing y...')
        hd = HourlyData('o3')
        hd.read_from_file()

        # time offset
        time_offset = settings.time_offset
        hd.df.index = pd.to_datetime(hd.df.index) + time_offset

        # flatten, but move through time steps first.
        y_values = hd.df.values.T.reshape(-1)
        y_df = pd.DataFrame(index=range(len(y_values)),
                            columns=['y'],
                            data=y_values)
        y_df.index.name = 'node_index'

        # save
        y_df.to_csv(self.y_path)
        print(f'written to {self.y_path}')

    def get_gaps(self):
        """
        Analyze the gaps in measurement data, their length.
        """
        print('analyzing existing gaps...')

        print('correlated gaps...')
        # read in data
        y_df_flat = pd.read_csv(self.y_path, index_col=0)
        reg_df = pd.read_csv(self.reg_path, index_col=0)
        reg_df.datetime = pd.to_datetime(reg_df.datetime)
        hd = HourlyData('o3')
        hd.read_from_file()
        y_df = hd.df
        y_df.index = pd.to_datetime(y_df.index) + settings.time_offset
        y_df.columns = [float(col) for col in y_df.columns]
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
        gap_type = 'missing_o3_mask'

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
                    print(f'{datetime} / {corr_series.index[-1]}')
                gap_len = 1
                datetime_gap_start = datetime
            missing_old = missing

        # set all values to 1 and all nans to 0
        y_df_flat.loc[y_df_flat.y==y_df_flat.y, 'y'] = 1
        y_df_flat.loc[y_df_flat.y!=y_df_flat.y, 'y'] = 0

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
                                              gap_len, gap_len]
                    gap_idx += 1
                y_index_gap_start = y_index
                gap_len = 1
            y_old = y
            station_id_old = station_id

        # save
        gap_df.to_csv(self.gap_path)
        print(f'written to {self.gap_path}')

    def get_mask(self):
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
        gap_df = pd.read_csv(self.gap_path, index_col=0)
        reg_df = pd.read_csv(self.reg_path, index_col=0)
        reg_df.datetime = pd.to_datetime(reg_df.datetime)
        y_df_flat = pd.read_csv(self.y_path, index_col=0)
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
        mask_df.index.name = 'node_index'

        # reset gap data frame
        gap_df = gap_df[gap_df.type=='missing_o3_mask'].copy()

        # missing o3 is simply where we do not have any measurement
        mask_df.loc[y_df_flat.y!=y_df_flat.y, 'missing_o3_mask'] = True
        
        
        # print summary statistics of the missing values
        n_total = len(mask_df)
        n_missing = mask_df.missing_o3_mask.sum()
        print('\nmissing measurements')
        print(f'{n_missing} of {n_total} = {n_missing/n_total*100:.1f} %')

        # station and time step info
        station_id_list = [float(id_) for id_ in y_df.columns]
        station_id_list = [int(id_) for id_ in station_id_list]
        n_stations = len(station_id_list)
        n_stations_without_data = y_df.isnull().all(axis=0).sum()
        n_stations_with_data = n_stations - n_stations_without_data
        n_timesteps = len(y_df)



        # analyze single gaps
        lens, counts = np.unique(gap_df[~gap_df.correlated].len,
                                 return_counts=True)
        print(f'\nsingle gaps')
        n_samples_gaps = gap_df[~gap_df.correlated].len.sum()
        for idx, len_ in enumerate(lens):
            n_gaps = counts[idx]
            n_samples = counts[idx] * len_
            print(f'length {len_}\t# gaps {n_gaps}\t# samples {n_samples}')
        print(f'\t\tsum: \t{n_samples_gaps} samples = ' +
              f'{n_samples_gaps/n_total*100:.1f} %')
        

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
        ratio = round(n_ratio / float(true_count), 2)

        print('\nshopping list each for val and test set')
        for idx, len_ in enumerate(lens):
            n_gaps_required = counts[idx] * ratio
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

        # now flag the gaps
        shopping_list.reverse()
            
            
        dis = 0  # tmp

        print('\nflagging single gaps...')
        correlated = False
        for len_, n_gaps_required in shopping_list:
            for mask in ['val_mask', 'test_mask']:
                n_gaps_found = 0
                while n_gaps_found < n_gaps_required:
                    # pick random start index of the gap
                    start_idx = random.choice(mask_df.index)
                    start_station = reg_df.loc[start_idx, 'station_id']
                    # add gap len to obtain end index
                    end_idx = start_idx + len_ - 1
                    # useless if it exceeds the original index
                    if end_idx > mask_df.index[-1]:
                        print('exceeds df boundaries')
                        continue
                    end_station = reg_df.loc[end_idx, 'station_id']
                    if start_station != end_station:
                        print('different station')
                        continue
                    # useless if any more than 10 % of the values True
                    mask_df_filtered = mask_df.loc[start_idx:end_idx]
                    missing_values = mask_df_filtered.sum().sum()
                    if missing_values/len_ > 0.1 :
                        print('too many missing values')
                        continue
                    
                    # set those to true that are not true elsewhere
                    index_list = mask_df_filtered[
                           (mask_df_filtered==False).all(axis=1)].index
                    mask_df.loc[index_list, mask] = True
                    n_samples = len(index_list)
                    start_datetime = reg_df.loc[start_idx, 'datetime']
                    end_datetime = reg_df.loc[end_idx, 'datetime']
                    _ = [start_station, mask, correlated,
                         start_datetime, end_datetime, start_idx,
                         len_, n_samples]
                    gap_df.loc[len(gap_df), :] = _
                    n_gaps_found += 1
                    print(f'found single {mask} gap of len {len_}' +
                          f' ({n_gaps_required-n_gaps_found} to go)')
        

        # train mask is where no other mask applies
        index_list = mask_df[(mask_df==False).all(axis=1)].index
        mask_df.loc[index_list, 'train_mask'] = True

        # convert gap df columns back to int
        gap_df = gap_df.astype({'station_id':int, 'start_idx':int,
                                'len':int, 'n_samples':int})

        # print a summary of all masks
        print('\nmask summary')
        n_samples = len(mask_df)
        print(f'{n_samples} in total ( 100.0 %)')
        for col in mask_df.columns:
            n_samples_set = mask_df[col].sum().sum()
            percentage = n_samples_set / n_samples * 100
            print(f'{n_samples_set} in {col} ( {percentage:.2f} %)')

        # mask sanity check
        assert mask_df.sum().sum() == len(mask_df)

        # save
        mask_df.to_csv(self.mask_path)
        print(f'\nwritten to {self.mask_path}')
        gap_df.to_csv(self.gap_path)
        print(f'written to {self.gap_path}')

    def get_edges_tmp(self):
        """
        Collect all edges. This routine best runs on the mem192
        partition of JUWELS.

        This routine also produces the edge_tmp.csv file,
        which makes calculating edge weights a lot faster.
        """
        print('preprocessing edges...')

        # settings for edges, radius in km and time window
        radius = settings.radius
        time_window = pd.Timedelta('0 days 00:00:00')

        # read in data
        station_df = pd.read_csv(self.station_path, index_col=0)
        reg_df = pd.read_csv(self.reg_path, index_col=0,
                             parse_dates=['datetime'])
        # reg_df = reg_df[0:int(1/1000*len(reg_df))].copy() # <-- for testing

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
        _, idx_lists = nn.radius_neighbors(coords,
                                           radius=radius)
        for trg_idx, src_idx_list in enumerate(idx_lists):
            trg_id = ids[trg_idx]
            src_ids = [ids[src_idx] for src_idx in sorted(src_idx_list)]
            neighbor_df.loc[trg_id, 'neighbors'] = src_ids

        # edges exist if measurements are at neighboring stations
        # and in the given time window
        reg_ddf = ddf.from_pandas(reg_df, npartitions=36)
        datetime_start = datetime.datetime.now()
        print(datetime_start)
        args = (neighbor_df, reg_df, radius, time_window)
        edge_dseries = reg_ddf.apply(neighbor_index_filter,
                                     args=args,
                                     axis=1,
                                     meta=edge_df['source_indices'])
        edge_series = edge_dseries.compute(scheduler='multiprocessing')
        edge_df.source_indices = edge_series
        print(datetime.datetime.now() - datetime_start)

        # save temporary edge_df
        edge_df.to_csv(self.edge_tmp_path)
        print(f'written to {self.edge_tmp_path}')

    
    
    def get_edges(self):
        edge_df = pd.read_csv(self.edge_tmp_path, index_col=0)
        source_index_list, target_index_list = [], []
        for index, row in edge_df.iterrows():
            source_indices = row.source_indices.strip('][').split(', ')
            source_indices = [int(idx) for idx in source_indices]
            target_index = index
            source_index_list += source_indices
            target_index_list += [target_index]*len(source_indices)
        edge_df = pd.DataFrame()
        edge_df['source_node_index'] = source_index_list
        edge_df['target_node_index'] = target_index_list
        edge_df.index.name = 'edge_index'
        edge_df.to_csv(self.edge_path)
        print(f'written to {self.edge_path}')

    def get_edge_weights(self):
        """
        Weight edges by spatial and temporal proximity
        """
        print('preprocessing edge weights...')

        # read in data
        edge_tmp_df = pd.read_csv(self.edge_tmp_path, index_col=0)
        pos_df = pd.read_csv(self.pos_path, index_col=0,
                             converters={'datetime': pd.Timestamp})

        # locations
        loc_df = pd.DataFrame()
        loc_df['x'], loc_df['y'], loc_df['z'] = geo_to_cartesian(
                                                pos_df.lon, pos_df.lat)

        # iterate through all node indices
        edge_weight_list = []
        for target_node_index in edge_tmp_df.index:
            if target_node_index % 50000 == 0:
                print(f'{target_node_index/len(edge_tmp_df)*100:.2f} %')
            source_node_indices = eval(edge_tmp_df.loc[target_node_index,
                                                      'source_indices'])
            loc_target_node = loc_df.loc[[target_node_index]]
            loc_source_nodes = loc_df.loc[source_node_indices]
            time_target_node = pos_df.loc[[target_node_index], 'datetime']
            time_source_nodes = pos_df.loc[source_node_indices, 'datetime']
            weights_source_nodes = loc_weight(loc_target_node, loc_source_nodes) * \
                                   time_weight(time_target_node, time_source_nodes)
            weights_source_nodes = weights_source_nodes/sum(weights_source_nodes)
            edge_weight_list += weights_source_nodes.tolist()

        # edge weight dataframe
        weight_df = pd.DataFrame(columns=['edge_weight'])
        weight_df['edge_weight'] = edge_weight_list
        weight_df.index.name = 'edge_index'

        # save
        weight_df.to_csv(self.weight_path)
        print(f'written to {self.weight_path}')

    def convert_to_tensor(self):
        """
        Convert all the .csv files to tensors
        """
        print('converting to pytorch tensors...')

        # x
        x_df = pd.read_csv(self.x_path, index_col=0)
        x_df = get_one_hot(x_df)
        x = torch.tensor(x_df.values.astype(np.float32))
        torch.save(x, self.x_pt_path)
        print(f'written to {self.x_pt_path}')
        del x_df

        # edges
        edge_df = pd.read_csv(self.edge_path, index_col=0)
        src_list = edge_df.source_node_index.to_list()
        trg_list = edge_df.target_node_index.to_list()
        edge_index = torch.tensor(np.array([src_list, trg_list])).detach()
        edge_index = edge_index.to(torch.long)
        torch.save(edge_index, self.edge_index_pt_path)
        print(f'written to {self.edge_index_pt_path}')
        del edge_df

        # edge weights
        weight_df = pd.read_csv(self.weight_path, index_col=0)
        edge_weights = torch.tensor(weight_df.values.astype(np.float32)).view(-1, 1)
        torch.save(edge_weights, self.edge_weight_pt_path)
        print(f'written to {self.edge_weight_pt_path}')
        del weight_df

        # y
        y_df = pd.read_csv(self.y_path, index_col=0)
        y = torch.tensor(y_df.values.astype(np.float32)).view(-1, 1)
        torch.save(y, self.y_pt_path)
        print(f'written to {self.y_pt_path}')
        del y_df

        # masks
        mask_df = pd.read_csv(self.mask_path, index_col=0)
        missing_o3_mask = torch.tensor(mask_df.missing_o3_mask).view(-1)
        torch.save(missing_o3_mask, self.missing_o3_mask_pt_path)
        print(f'written to {self.missing_o3_mask_pt_path}')
        train_mask = torch.tensor(mask_df.train_mask).view(-1)
        torch.save(train_mask, self.train_mask_pt_path)
        print(f'written to {self.train_mask_pt_path}')
        val_mask = torch.tensor(mask_df.val_mask).view(-1)
        torch.save(val_mask, self.val_mask_pt_path)
        print(f'written to {self.val_mask_pt_path}')
        test_mask = torch.tensor(mask_df.test_mask).view(-1)
        torch.save(test_mask, self.test_mask_pt_path)
        print(f'written to {self.test_mask_pt_path}')
        del mask_df

        # position
        pos_df = pd.read_csv(self.pos_path, index_col=0)
        pos = torch.tensor(pos_df[['lon', 'lat']].values.astype(np.float32))
        torch.save(pos, self.pos_pt_path)
        print(f'written to {self.pos_pt_path}')
        del pos_df

    def get_dataset(self):
        """
        Finalize the Pytorch dataset, include everything that is
        in the gAQBench dataset.
        """
        # read in data
        x = torch.load(self.x_pt_path)
        edge_index = torch.load(self.edge_index_pt_path)
        edge_weight = torch.load(self.edge_weight_pt_path)
        y = torch.load(self.y_pt_path)
        missing_o3_mask = torch.load(self.missing_o3_mask_pt_path)
        train_mask = torch.load(self.train_mask_pt_path)
        val_mask = torch.load(self.val_mask_pt_path)
        test_mask = torch.load(self.test_mask_pt_path)
        pos = torch.load(self.pos_pt_path)

        # create dataset
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                    y=y, missing_o3_mask=missing_o3_mask,
                    train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask, pos=pos)
        self.data, self.slices = self.collate([data])


def dataset_workflow():
    """
    Full workflow to create the dataset
    """
    ug = UBAGraph()
    #ug.get_reg()
    #ug.get_pos()
    #ug.get_x()
    ug.get_y()
    #ug.get_gaps()
    #ug.get_mask()
    #ug.get_edges_tmp()
    #ug.get_edges()
    #ug.get_edge_weights()
    #ug.convert_to_tensor()


def print_graph_statistics():
    """
    Read in dataset and print basic statistics
    """

    ug = UBAGraph()
    ug.get_dataset()
    dataset = ug

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of node features: {dataset.num_features}')
    print(f'Number of edge features: {dataset.num_edge_features}')
    data = dataset[0]  # Get the first graph object.
    print(data)
    print('======================')
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {(2*data.num_edges) / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node share: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


if __name__ == '__main__':
    """
    Start routines
    """
    dataset_workflow()
    #print_graph_statistics()
