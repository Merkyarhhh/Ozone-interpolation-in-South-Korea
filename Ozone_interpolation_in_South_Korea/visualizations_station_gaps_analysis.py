"""
An old script for analyzing gap lenghts. Not meant for reuse.
"""

# general
import random
import pdb
import warnings
import pickle as pkl

# data science
import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# pytorch
from torch_geometric.data import Dataset

# plotting
import matplotlib.pyplot as plt
import networkx as nx

# own package
import settings
from retrieval_toar_db import StationData
from retrieval_toar_db import HourlyData

# oppress future warnings
warnings.filterwarnings('ignore')

class StnGapAnalysis():
    """ This analyses the distribution of gaps for differernt station types """
    def __init__(self):
        ### LOAD RAW DATASETS ###
        self.stn_data = pd.read_csv(settings.resources_dir + 'time_resolved_raw/' +
                          'stations.csv', index_col=0)
        self.gap_data = pd.read_csv(settings.resources_dir + 'time_resolved_preproc/' +
                               'gap_2011.csv', index_col=0)  ## Read only 2011 data
        self.reg_data = pd.read_csv(settings.resources_dir + 'time_resolved_preproc/' +
                               'reg.csv', index_col=0)
        print(f'Loaded station data and gap data......')

    def select_dates(self):
        print(f' No. of nodes in total: {len(self.reg_data)}')
        data_2011 = self.reg_data[self.reg_data['datetime'].str.contains('2011')] # Select the year 2011
        print(f'data in 2011: {data_2011.head()}')
        print(f'data in 2011: {data_2011.tail()}')
        self.id_2011 = data_2011.index.astype('float')
        print(f' Node index for 2011: {self.id_2011[:5]}')
        print(f' Node index for 2011: {self.id_2011[-1]}')
        print(f' No. of nodes in 2011: {len(self.id_2011)}')

    def define_large_gap(self):
      ### DEFINE LARGE GAPS AND GAP BINS ###
        real_gap = self.gap_data[(self.gap_data['type']=='missing_o3')]
        print(f'No. of real gaps in total: {len(real_gap)}')

        self.large_gap = real_gap[real_gap['len'] > 12.0] # Define large gaps as gaps longer than 12.0 hours
        #self.large_gap = real_gap_2011[real_gap_2011['len'] > 12.0] # Select gaps only in 2011
        print(f'No. of large gaps: {len(self.large_gap)}')
        # Define gap bin = [12h, 1d, 2d, 3d, 1w, 2w, 1m, 2m, 3m, 6m, 1y 2y]
        self.gbin = [12, 24, 48, 72, 168, 336, 720, 1440, 2160, 4320, 8760, 17520, 43824] # define bin for gap length

    def define_gap_bins(self):
        real_gap = self.gap_data[(self.gap_data['type']=='missing_o3')]
        self.large_gap = real_gap ## this includes all gaps, not only "large" gap, but just stick to the terminology

        # Define gap bin = [1h, 2h, 3h, 4h, 6h, 12h, 1d, 2d, 3d, 1w, 2w, 1m, 2m, 3m, 6m, 1y]
        self.gbin = [1, 2, 3, 4, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2160, 4320, 8760, 17520] # define bin for gap length

    def large_gap_hist(self):
        ### PLOT GAP BIN HISTOGRAM ###
        glen = self.large_gap['len']
        hist = glen.hist(bins=self.gbin)
        plt.gca().set_yscale("log")
        plt.gca().set_xscale("log")
        plt.xlabel('gap length (h)')
        plt.ylabel('count')
        plt.title('gap distribution in length in bins')
        plt.savefig(settings.output_dir + 'gap_len_hist_bins.png')
        plt.clf()
        print(f'Number of gaps per bin: { glen.value_counts(bins=self.gbin) }')

    def stn_typs_hist(self):
        ### Analysis station data by station types ###
        # Print counts of stations per type per type of area
        stn_typs = self.stn_data[['type','type_of_area']].value_counts()
        print(stn_typs)
        # Check missing values
        stn_typ = self.stn_data['type']
        stn_atyp = self.stn_data['type_of_area']
        print(f'Number of missing values in type: { stn_typ.isna().sum() } ')
        print(f'Number of missing values in type of area: { stn_atyp.isna().sum() }')

        # Plot histogram
        pd.crosstab(self.stn_data['type'],self.stn_data['type_of_area']).plot.bar()
        plt.title("Station types and type of area")
        plt.xlabel("type")
        plt.xticks(rotation=0)
        plt.ylabel("Number of stations")
        plt.savefig(settings.output_dir + 'station_types.png')
        plt.clf()


    def stn_typs_gap_count(self):
        ### Count gaps per bin in each station ###
        # Classify gaps to gap bin (group)
        labels = ["Gap_{0}".format(i) for i in range(len(self.gbin)-1)]
        self.large_gap['group'] = pd.cut(self.large_gap.len, self.gbin, right=False, labels=labels)
        print(self.large_gap.head())
        #self.large_gap.to_csv(settings.output_dir + 'gap_tobins.csv')
        # Count no. of gaps per bin in each station
        stn_ggroup = pd.crosstab(self.large_gap['station_id'],self.large_gap['group'])
        print(stn_ggroup.head())

        # Save to csv output
        stn_ggroup.to_csv(settings.output_dir + 'station_gap_length.csv')
        print(f'written to csv file station_gap_length.csv......')

        ### Merging gap group table and station table ###
        stn_ggroup_data = pd.read_csv(settings.output_dir +'station_gap_length.csv')
        stn_ggroup_new = stn_ggroup_data.rename(columns={'station_id': 'id'})
        stn_data_ggroup = pd.merge(self.stn_data, stn_ggroup_new, how='left', on='id')
        print(stn_data_ggroup.head())

        #Save to csv output
        stn_data_ggroup.to_csv(settings.output_dir + 'station_fullinfo_gap_length.csv')
        print(f'written to csv file station_fullinfo_gap_length.csv......')

        ### Analyse distribution of station types per gap bin ###
        for i in range(len(self.gbin)-1):
            gap_name = "Gap_{0}".format(i)
            #Count no. of gaps per bin
            print(f'Number of gaps in {gap_name}: {stn_data_ggroup[gap_name].sum()} ')
            # Count no. of gaps per bin per station type
            stn_typ_gap = stn_data_ggroup.groupby(['type','type_of_area'])[gap_name].sum()
            print(stn_typ_gap)

            # Plot histogram
            stn_typ_gap.unstack().plot.bar()
            plt.title(f"Gap {i}")
            plt.xlabel("type")
            plt.xticks(rotation=0)
            plt.ylabel("Number of gaps")
            gn_str = str(i).zfill(2)
            plt.savefig(settings.output_dir + f"station_types_gap"+gn_str+".png")


if __name__ == '__main__':
    """
    Create the dataset for testing purposes.
    """
    sga = StnGapAnalysis()
    #sga.select_dates()
    #sga.define_large_gap()
    sga.define_gap_bins()
    sga.large_gap_hist()
    sga.stn_typs_hist()
    sga.stn_typs_gap_count()



