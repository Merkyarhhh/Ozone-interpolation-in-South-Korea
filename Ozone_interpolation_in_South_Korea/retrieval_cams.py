"""
This file contains all routines that were used in retrieving and
preprocessing CAMS data. They are meant for reference, not for reuse.

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

class GetInterpolatedCAMSData():
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
        
    def download_cams_data(self):
        ### This requires a cdsapi userid and keys ###
        outfile = self.cams_dir + 'cams_eac4_2021_3hr.nc'

        c = cdsapi.Client(timeout=1200,quiet=False,debug=True)

        c.retrieve(
            'cams-global-reanalysis-eac4',
            {
                'variable': ['nitrogen_dioxide', 'nitrogen_monoxide', 'ozone',],
                'model_level': '60',
                'date': '2021-01-01/2021-12-31',
                'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                        ],
                'area': [
                        140, 22, 100, 53,
                        ],
                'format': 'netcdf',
            },
            outfile
            )    
    
    def read_eac4_ncfile(self):
        # Convert unit from mmr to vmr (ppb)
        no_unit = 28.9644/30.0061*1.e9
        no2_unit = 28.9644/46.0055*1.e9 
        o3_unit = 28.9644/47.9982*1.e9
        so2_unit = 28.9644/64.0638*1.e9
        co_unit = 28.9644/28.0101*1.e9
        
        ## READ nc file
        eac4_path = self.cams_dir + 'levtype_ml.nc'
        nc_data = xr.open_dataset(eac4_path)
        self.no = nc_data['no'][:,:,:] * no_unit 
        self.no2 = nc_data['no2'][:,:,:] * no2_unit 
        self.o3 = nc_data['go3'][:,:,:] * o3_unit 
        self.so2 = nc_data['so2'][:,:,:] * so2_unit 
        self.co = nc_data['co'][:,:,:] * co_unit 
        self.lati  = nc_data['latitude'][:]                       
        self.loni  = nc_data['longitude'][:]
        self.ti = nc_data['time'][:]
        self.nt = len(self.ti)
    
    def read_pm_ncfile(self):
        ## READ nc file
        emiss_path = self.cams_dir + 'levtype_sfc.nc'
        nc_data = xr.open_dataset(emiss_path)
        self.pm10 = nc_data['pm10'][:,:,:] * 1000000000
        self.pm25 = nc_data['pm2p5'][:,:,:] * 1000000000
        self.late  = nc_data['latitude'][:]                       
        self.lone  = nc_data['longitude'][:]
        self.te = nc_data['time'][:]
        self.nte = len(self.te)
        
    def read_stn_info(self):
        stn_data = pd.read_csv(self.stn_path)
        self.ids = stn_data['id']
        self.lats = stn_data['lat']
        self.lons = stn_data['lon']
        self.ns = len(self.ids)
        print(f' No. of stations: {self.ns}')     
        print(f' Station IDs:{self.ids}')
        
    def inp_latlon(self,var,data,time_index,lats,lons):
            var_str = var
            nt,nlat,nlon = np.shape(data)
            ns = self.ns  # station dimension
        
            # Create empty station interpolated arrays 
            data_stn = np.zeros(nt*ns) 
            data_stn = data_stn.reshape(nt,ns)
            
            # Select data by interpolating the neareast lat & lon grids
            for i in range(ns):
            #for i in range (0,5):
                id = self.ids[i]
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
            id_str = self.ids.to_numpy(dtype=str)
            stn_df.columns = id_str
            stn_df['time'] = time_index
            stn_df = stn_df.set_index('time')
                
            # Resample and interpolate data to one-hour interval
            stn_df_2021 = stn_df['2021']
            last_hour = pd.to_datetime('2021-12-31 23:00:00')
            stn_df_2021 = stn_df_2021.append(pd.DataFrame(index=[last_hour]))
            stn_df_1h = stn_df_2021.resample('60T')
            stn_df_1h = stn_df_1h.interpolate(method='linear',axis=0) 
            
            # Save as csv file
            out_csv_file = self.cams_dir + 'hourly_cams_' + var_str + '.csv' 
            stn_df_1h.to_csv(out_csv_file)
    
    def get_eac4_data_latlon(self):
        self.inp_latlon("no",self.no,self.ti,self.lats,self.lons)
        self.inp_latlon("no2",self.no2,self.ti,self.lats,self.lons)
        self.inp_latlon("o3",self.o3,self.ti,self.lats,self.lons)
        self.inp_latlon("so2",self.so2,self.ti,self.lats,self.lons)
        self.inp_latlon("co",self.co,self.ti,self.lats,self.lons)
        
    
    def get_pm_data_latlon(self):
        self.inp_latlon("pm10",self.pm10,self.te,self.lats,self.lons)
        self.inp_latlon("pm25",self.pm25,self.te,self.lats,self.lons)



if __name__ == '__main__':
    """
    Create the dataset for testing purposes.
    """
    gicd = GetInterpolatedCAMSData()
    ## DOWNLOAD CAMS files # need to modify code above #
    gicd.download_cams_data()
    ## READ station file
    #gicd.read_stn_info()
    
    ## For reanalysis data
    ## READ_ncfile
    #gicd.read_eac4_ncfile()
    ## GET station-interpolated data
    #gicd.get_eac4_data_latlon()
    
    ## For pm data
    ## READ ncfile
    #gicd.read_pm_ncfile()
    ## GET station-interpolated data
    #gicd.get_pm_data_latlon()


