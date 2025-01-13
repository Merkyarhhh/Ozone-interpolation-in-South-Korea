"""
This script preprocesses raw cams data to .csv files. These files
are later used in uba.py to create the graph dataset.
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

# own
import settings

class GetInterpolatedCAMSData():
    """
    Get dataset from CAMS global reanalysis data (EAC4) interpolated per stations and nodes
    """
    def __init__(self):
        """
        Initialize the class
        """
        # paths
        self.in_dir = settings.resources_dir + 'time_resolved_raw/'
        self.out_dir = settings.resources_dir + 'time_resolved_raw/'
        self.stn_path = self.in_dir + 'stations.csv'
        #self.nc_path = self.in_dir + 'cams_eac4_2011_3hourly.nc'
        self.reg_path = self.out_dir + 'reg.csv'
        self.x_path = self.out_dir + 'x.csv'
        self.csv_out_path = self.in_dir + 'cams_data_csv/'
        self.plot_out_path = settings.output_dir
        print(f'Resource path: {self.in_dir}')


    def read_eac4_ncfile(self):

        # Convert unit from mmr to vmr (ppb)
        no_unit = 28.9644/30.0061*1.e9
        no2_unit = 28.9644/46.0055*1.e9
        o3_unit = 28.9644/47.9982*1.e9

        ## READ nc file
        eac4_path = self.in_dir + 'cams_eac4_2011_3hourly.nc'
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
        emiss_path = self.in_dir + 'CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nox_v5.3_monthly_2011_DE.nc'
        nc_data = xr.open_dataset(emiss_path)
        esum_nox = nc_data['sum'][:,:,:]
        self.late  = nc_data['lat'][:]
        self.lone  = nc_data['lon'][:]
        self.te = nc_data['time'][:]
        self.nte = len(self.te)
        self.emiss_nox = esum_nox * 12.0 * 1.e12 / 100.0 / 1.e6 ## Convert unit - from Tg per month to g m−2 yr−1
        ## Print arrays to check

    def read_stn_info(self):
        stn_data = pd.read_csv(self.stn_path)
        self.ids = stn_data['id']
        self.lats = stn_data['lat']
        self.lons = stn_data['lon']
        self.ns = len(self.ids)
        print(f' No. of stations: {self.ns}')
        print(f' Station IDs:{self.ids}')

    def plot_ncfile(self):
        def ymean_plot(varname, lons, lats, x, y, data):
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
            m =Basemap(projection='merc',llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,urcrnrlat=maxlat,resolution='i')
            m.drawcountries()
            m.drawcoastlines()
            # Plot color mesh
            m.pcolormesh(lon, lat, ymean, latlon=True, cmap='RdBu_r')
            plt.colorbar(label= vstr+' VMR (ppb)')
            # add stations
            xs,ys = m(lons,lats)
            m.plot(xs,ys,'ro',markersize=2)
            plt.savefig(self.plot_out_path + 'cmap_'+vstr+'_2011avg.png')
            plt.clf()

        ymean_plot('no',self.lons,self.lats,self.loni, self.lati, self.no)
        print(f'Plotted NO color map...')
        ymean_plot('no2',self.lons,self.lats,self.loni, self.lati,self.no2)
        print(f'Plotted NO2 color map...')
        ymean_plot('o3',self.lons,self.lats,self.loni, self.lati,self.o3)
        print(f'Plotted O3 color map...')

    def get_data_latlon(self):
        def inp_latlon(var,data,time_index,lats,lons,ids):
            var_str = var
            nt,nlat,nlon = np.shape(data)
            ns = self.ns  # station dimension

            # Create empty station interpolated arrays
            data_stn = np.zeros(nt*ns)
            data_stn = data_stn.reshape(nt,ns)

            # Select data by interpolating the neareast lat & lon grids
            for i in range(ns):
            #for i in range (0,5):
                id = ids[i]
                lat = lats[i]
                lon = lons[i]
                print (f'Interpolating Station {id} ({i+1}/{ns})...')
                print (f'lat & lon : ({lat},{lon})')

                for t in range(nt):
                    try:
                        data_stn[t,i]=data[t,:,:].interp(latitude=lat, longitude=lon)
                    except:
                        data_stn[t,i]=data[t,:,:].interp(lat=lat, lon=lon)
            print(f'Interpolated data: {data_stn}')

            # Convert to panda dataframe
            stn_df = pd.DataFrame(data_stn)
            id_str = ids.to_numpy(dtype=str)
            print(f'id_str: {id_str}')
            stn_df.columns = id_str
            print(f'Time index: {time_index}')
            stn_df['time'] = time_index
            stn_df = stn_df.set_index('time')
            print(f'station data : {stn_df.head()}')

            # Resample and interpolate data to one-hour interval
            stn_df_2011 = stn_df['2011']
            last_hour = pd.to_datetime('2011-12-31 23:00:00')
            stn_df_2011 = stn_df_2011.append(pd.DataFrame(index=[last_hour]))
            stn_df_1h = stn_df_2011.resample('60T')
            stn_df_1h = stn_df_1h.interpolate(method='linear',axis=0)
            #stn_df_1h = stn_df_1h.interpolate(method='spline',order=2,axis=0)
            print(f'station 1h data : {stn_df_1h.head()}')

            # Save as csv file
            out_csv_file = self.out_dir + 'hourly_cams_' + var_str + '.csv'
            stn_df_1h.to_csv(out_csv_file)

        inp_latlon("no",self.no,self.ti,self.lats,self.lons,self.ids)
        inp_latlon("no2",self.no2,self.ti,self.lats,self.lons,self.ids)
        inp_latlon("o3",self.o3,self.ti,self.lats,self.lons,self.ids)
        inp_latlon("Enox",self.emiss_nox,self.te,self.lats,self.lons,self.ids)

    def stn_data_stats(self):
        def plot_conc_hist(varname, data, binwidth):
            vstr = varname
            maxvmr = int(np.max(data))
            ppb_bin = range(0,maxvmr,binwidth)
            hist = data.hist(bins=ppb_bin)
            plt.gca().set_yscale("log")
            #plt.gca().set_xscale("log")
            plt.title('CAMS '+vstr+' distribution')
            plt.xlabel('VMR (ppb)')
            plt.ylabel('count')
            plt.savefig(settings.output_dir + 'cams_'+vstr+'_2011_hist.png')
            plt.clf()
        # Present conc arrays
        m = 365*24 + 1 # read one year of data in 1-hour interval
        n = self.ns  # station dimension

        no_stn = np.zeros(n*m)
        no_stn = no_stn.reshape(n,m)
        no2_stn = np.zeros(n*m)
        no2_stn = no2_stn.reshape(n,m)
        o3_stn = np.zeros(n*m)
        o3_stn = o3_stn.reshape(n,m)
        # READ station interpolated data from precompiled csv files
        for i in range(n):
            id = self.ids[i]
            id_str = str(id).zfill(4)
            data_stn = pd.read_csv(self.csv_out_path + 'ID_' + id_str + '_1h.csv',index_col=0)
            no_stn[i,:] = data_stn['no']
            no2_stn[i,:] = data_stn['no2']
            o3_stn[i,:] = data_stn['o3']
            print(f'Reading csv file for Station {id} ({i+1}/{n})...')
            # Print overall statistics
        no_df = pd.DataFrame(no_stn.flatten())
        no2_df = pd.DataFrame(no2_stn.flatten())
        o3_df = pd.DataFrame(o3_stn.flatten())
        print(no_df.describe())
        print(no2_df.describe())
        print(o3_df.describe())
        # Plot distribution histogram
        plot_conc_hist('no',no_df,10)
        plot_conc_hist('no2',no2_df,5)
        plot_conc_hist('o3',o3_df,5)



if __name__ == '__main__':
    """
    Create the dataset for testing purposes.
    """
    gicd = GetInterpolatedCAMSData()
    ## READ ncfile
    gicd.read_eac4_ncfile()
    gicd.read_emiss_ncfile()
    ## READ station file
    gicd.read_stn_info()
    ## PLOT colour maps
    #gied.plot_ncfile()
    ## GET station-interpolated data
    gicd.get_data_latlon()
    ## OUTPUT basic interpolated data statistics
    #gicd.stn_data_stats()

