"""
This file contains routines that were used in visualise and
pre-analyse the CAMS data.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import netCDF4 as nc   # Reading NetCDF4 files.
import time
import json
import csv
import xarray as xr

# for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# for cams cds
import cdsapi

# own
import settings

class PlotCAMSData():
    """
    Get dataset from CAMS global reanalysis data (EAC4) interpolated per stations and nodes
    """
    def __init__(self):
        """
        Initialize the class
        """
        # paths
        self.cams_dir = settings.resources_dir + 'uba_graph_raw/cams/'
        self.stn_path = settings.resources_dir + 'uba_graph_raw/' + 'stations.csv'
        self.plot_out_dir = settings.output_dir
        self.hourly_no_path = self.cams_dir + 'hourly_cams_no.csv'
        self.hourly_no2_path = self.cams_dir + 'hourly_cams_no2.csv'
        self.hourly_o3_path = self.cams_dir + 'hourly_cams_o3.csv'
        self.hourly_Enox_path = self.cams_dir + 'hourly_cams_Enox.csv'

    def read_eac4_ncfile(self):
        # Convert unit from mmr to vmr (ppb)
        no_unit = 28.9644/30.0061*1.e9
        no2_unit = 28.9644/46.0055*1.e9
        o3_unit = 28.9644/47.9982*1.e9

        ## READ nc file
        eac4_path = self.cams_dir + 'cams_eac4_2011_3hourly.nc'
        nc_data = xr.open_dataset(eac4_path)
        self.no = nc_data['no'][:,:,:] * no_unit
        self.no2 = nc_data['no2'][:,:,:] * no2_unit
        self.o3 = nc_data['go3'][:,:,:] * o3_unit
        self.lati  = nc_data['latitude'][:]
        self.loni  = nc_data['longitude'][:]
        self.ti = nc_data['time'][:]
        self.nt = len(self.ti)

    def read_emiss_ncfile(self):
        ## READ nc file
        emiss_path = self.cams_dir + 'CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nox_v5.3_monthly_2011_DE.nc'
        nc_data = xr.open_dataset(emiss_path)
        esum_nox = nc_data['sum'][:,:,:]
        self.late  = nc_data['lat'][:]
        self.lone  = nc_data['lon'][:]
        self.te = nc_data['time'][:]
        self.nte = len(self.te)
        self.emiss_nox = esum_nox * 12.0 * 1.e12 / 100.0 / 1.e6 ## Convert unit - from Tg per month to g m−2 yr−1

    def read_stn_info(self):
        stn_data = pd.read_csv(self.stn_path)
        self.ids = stn_data['id']
        self.lats = stn_data['lat']
        self.lons = stn_data['lon']
        self.ns = len(self.ids)
        print(f' No. of stations: {self.ns}')
        print(f' Station IDs:{self.ids}')

    def ymean_plot(self,varname, lons, lats, x, y, data):
        vstr=varname
        lon = x
        lat = y
        minlon = np.min(lon)
        maxlon = np.max(lon)
        minlat = np.min(lat)
        maxlat = np.max(lat)
        # make color plots with station locations
        lon, lat = np.meshgrid(lon, lat)
        fig = plt.figure(figsize=(8, 8))
        ymean = data.mean(dim='time')
        # Plot map lines
        m=Basemap(
            projection='merc',llcrnrlon=minlon,llcrnrlat=minlat,
            urcrnrlon=maxlon,urcrnrlat=maxlat,resolution='i'
            )
        m.drawcountries()
        m.drawcoastlines()
        # Plot color mesh
        m.pcolormesh(lon, lat, ymean, latlon=True, cmap='RdBu_r')
        if (vstr == "Enox"):
            plt.colorbar(label= vstr+'(g m−2 yr−1)')
        else:
            plt.colorbar(label= vstr+' VMR (ppb)')

        # add stations
        xs,ys = m(lons,lats)
        m.plot(xs,ys,'ro',markersize=2)
        plt.savefig(self.plot_out_dir + 'cmap_'+vstr+'_2011avg.png')
        plt.clf()

    def plot_hist(self,varname, data, binwidth):
        vstr = varname
        maxvmr = data.max(axis=0)
        ppb_bin = range(0,int(maxvmr[0]),binwidth)
        hist = data.hist(bins=ppb_bin)
        plt.gca().set_yscale("log")
        #plt.gca().set_xscale("log")
        plt.title('CAMS '+vstr+' distribution')
        if (vstr == "Enox"):
            plt.xlabel('NOx emissions (g m−2 yr−1)')
        else:
            plt.xlabel('VMR (ppb)')
        plt.ylabel('count')
        plt.savefig(self.plot_out_dir + 'cams_'+vstr+'_2011_hist.png')
        plt.clf()

    def plot_maps(self):
        self.ymean_plot('no',self.lons,self.lats,self.loni, self.lati, self.no)
        print(f'Plotted NO color map...')
        self.ymean_plot('no2',self.lons,self.lats,self.loni, self.lati,self.no2)
        print(f'Plotted NO2 color map...')
        self.ymean_plot('o3',self.lons,self.lats,self.loni, self.lati,self.o3)
        print(f'Plotted O3 color map...')
        self.ymean_plot('Enox',self.lons,self.lats,self.lone, self.late,self.emiss_nox)
        print(f'Plotted Enox color map...')


    def plot_houly_hist_stats(self):
        self.hourly_no = pd.read_csv(self.hourly_no_path,index_col=0)
        self.hourly_no2 = pd.read_csv(self.hourly_no2_path,index_col=0)
        self.hourly_o3 = pd.read_csv(self.hourly_o3_path,index_col=0)
        self.hourly_Enox = pd.read_csv(self.hourly_Enox_path,index_col=0)

        no_df = pd.DataFrame(self.hourly_no.to_numpy().flatten())
        no2_df = pd.DataFrame(self.hourly_no2.to_numpy().flatten())
        o3_df = pd.DataFrame(self.hourly_o3.to_numpy().flatten())
        Enox_df = pd.DataFrame(self.hourly_Enox.to_numpy().flatten())

        self.plot_hist('no',no_df,10)
        self.plot_hist('no2',no2_df,5)
        self.plot_hist('o3',o3_df,5)
        self.plot_hist('Enox',Enox_df,10)

        print(no_df.describe())
        print(no2_df.describe())
        print(o3_df.describe())
        print(Enox_df.describe())

if __name__ == '__main__':
    """
    Create the dataset for testing purposes.
    """
    pcd = PlotCAMSData()
    ## READ ncfile
    pcd.read_eac4_ncfile()
    pcd.read_emiss_ncfile()
    ## READ station info
    pcd.read_stn_info()
    ## PLOT colour maps
    pcd.plot_maps()
    ## OUPUT basic interpolated data histogram and statistics
    pcd.plot_houly_hist_stats()

