# general
import pdb

# data science
import numpy as np
import pandas as pd
import geopandas

# sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

# torch
import torch

# plotting
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.io import shapereader

# own package
import settings
from retrieval_toar_db import StationData, HourlyData
from preprocessing_uba import UBAGraph
from preprocessing_utils import geo_to_cartesian

from postprocessing_evaluation_tools import count_exceedances
from postprocessing_evaluation_tools import number_of_neighbors
from postprocessing_evaluation_tools import get_characteristics_df
from postprocessing_evaluation_tools import index_of_agreement
from visualizations_utils import scale_bar


def station_loc_on_map():
    """
    To give the readers an idea how the network looks like
    """
    print('station locations on map...')

    # projection
    extent = [125,131,33,39]
    #extent = [126,128,37,38]
    # Projection = ccrs.Orthographic(
    #                  central_longitude=(0.5*(extent[0]+extent[1])),
    #                  central_latitude=(0.5*(extent[2]+extent[3])))
    # crs_proj4 = Projection.proj4_init
    Projection = ccrs.PlateCarree()

    # get country borders
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'
    #name = 'admin_1_states_provinces'
    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = geopandas.read_file(shpfilename)
    df_de = df.loc[df['ADMIN'] == 'South Korea']
    #df_de = df.loc[df['name'] == 'Seoul']
    df_de.crs = 'EPSG:4326'
    # df_ae = df_de.to_crs(crs_proj4)

    # prepare ozone data
    hd = HourlyData('o3')
    hd.read_from_file()
    o3_df = hd.df
    null_cols = o3_df.columns[o3_df.isnull().all()]
    o3_df.drop(null_cols, axis=1, inplace=True)
    null_cols = [int(null_col) for null_col in null_cols]
    o3_mean_list = o3_df.mean().to_list()

    # prepare station data
    sd = StationData()
    sd.read_from_file()
    station_df = sd.df
    #station_df.drop(null_cols, axis=0, inplace=True)

    # nearest neighbors
    max_dist = 23
    x_, y_, z_ = geo_to_cartesian(station_df.lon.values,
                                  station_df.lat.values)
    coords = np.array([x_, y_, z_]).T
    nn = NearestNeighbors()
    nn.fit(coords)
    dist_lists, idx_lists = nn.radius_neighbors(coords,
                                                radius=max_dist)

    # prepare data for plotting
    df = pd.DataFrame()
    df['lon'] = station_df.lon
    df['lat'] = station_df.lat
    df['o3'] = o3_mean_list
    df['neighbors'] = idx_lists
    df['distances'] = dist_lists
    df.reset_index(inplace=True)
    vmin = np.percentile(df.o3, 5)
    vmax = np.percentile(df.o3, 95)

    # set up plot
    fig, ax = plt.subplots(subplot_kw={'projection': Projection})
    ax.set_extent(extent)
    ax.add_geometries(
                      # df_ae['geometry'],
                      df_de['geometry'],
                      crs=Projection,
                      # facecolor='gainsboro',
                      facecolor='white',
                      edgecolor='black', lw=0.1,
                      zorder=1, alpha=.8)
    # scale_bar(ax, 100)

    # geopandas (tmp)
    # gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
    # gdf.plot(ax=ax, color='red')
    # gdf.crs = 'EPSG:4326'
    # pdb.set_trace()
    # gdf = gdf.to_crs(crs_proj4)

    # nodes
    ax.scatter(x=df.lon,
               y=df.lat,
               c=df.o3,
               marker='o',
               linewidths=.2,
               cmap='coolwarm',
               s=5,
               zorder=3,
               edgecolor='black',
               vmin=vmin,
               vmax=vmax)
    ax.axis('off')
    fig = plt.gcf()

    # edges
    for node_1 in df.index:
        lon_1 = df.lon[node_1]
        lat_1 = df.lat[node_1]
        for idx_2, node_2 in enumerate(df.neighbors[node_1]):
            dist = df.distances[node_1][idx_2]
            alpha = .9*(max_dist-dist)/max_dist
            lon_2 = df.lon[node_2]
            lat_2 = df.lat[node_2]
            plt.plot([lon_1, lon_2],
                     [lat_1, lat_2],
                     zorder=2,
                     color='k',
                     lw=.5,
                     alpha=alpha)

    # save plot
    save_path = settings.output_dir + 'station_loc_on_map_germany.png'
    plt.savefig(save_path, dpi=3000, bbox_inches='tight', pad_inches=0.)
    print(f'plot saved to {save_path}')
    plt.close()

    # plot colorbar
    fig, ax = plt.subplots(1, 1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm),
                              ax=ax,
                              pad=.05,
                              fraction=1,
                              cmap='coolwarm',
                              extend='both')
    cbar.set_label('Mean $O_3$ (ppbv)')
    ax.axis('off')

    # save colorbar
    save_path = settings.output_dir + 'station_loc_on_map_colorbar_germany.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
    print(f'saved to {save_path}')
    plt.close()


def station_ids_on_map():
    """
    To analyze what the station ids mean.
    This plot was not published.
    """
    print('station ids on map...')

    # get country borders
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'
    #name = 'admin_1_states_provinces'
    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = geopandas.read_file(shpfilename)
    poly = df.loc[df['ADMIN'] == 'South Korea']['geometry'].values[0]
    #poly = df.loc[df['name'] == 'Seoul']['geometry'].values[0]

    # prepare data
    sd = StationData()
    sd.read_from_file()
    station_df = sd.df
    station_df.reset_index(inplace=True)

    # plot
    ax = plt.axes(projection=ccrs.Orthographic())
    ax.add_geometries(poly,
                      crs=ccrs.Orthographic(),
                      facecolor='gainsboro',
                      edgecolor='black', lw=0.,
                      zorder=1, alpha=.75)

    ax.set_extent([125,128,35.5,37.5], crs=ccrs.Orthographic())
    #ax.set_extent([126,128,37,38], crs=ccrs.Orthographic())

    ax.scatter(x=station_df.lon,
               y=station_df.lat,
               c=station_df.index,
               marker='o',
               linewidths=.2,
               s=5,
               zorder=3,
               cmap='Blues',
               edgecolor='black')
    ax.axis('off')
    fig = plt.gcf()

    # save plot
    save_path = settings.output_dir + 'stations_on_map_bjd.png'
    plt.savefig(save_path, dpi=3000, bbox_inches='tight', pad_inches=0.)
    print(f'plot saved to {save_path}')
    plt.close()

    # plot colorbar
    fig, ax = plt.subplots(1, 1)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(station_df))
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap='Blues'),
                ax=ax, pad=.05, fraction=1, orientation='horizontal')
    cbar.set_ticks([0, len(station_df)])
    cbar.set_ticklabels(['low', 'high'])
    cbar.set_label('station id')
    ax.axis('off')

    # save colorbar
    save_path = settings.output_dir + 'stations_on_map_colorbar_bjd.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
    print(f'saved to {save_path}')
    plt.close()


def o3_value_matrix():
    """
    All o3 measurements we have
    """
    print('o3 value matrix...')

    # read data
    hd = HourlyData('o3')
    hd.read_from_file()
    data = hd.df.values.T
    vmin = 0.
    vmax = np.nanpercentile(data, 99)

    # plot
    im = plt.imshow(data, interpolation='none', vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.axis('off')

    # save figure
    save_path = settings.output_dir + 'o3_value_matrix.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=2000)
    print(f'saved to {save_path}')
    plt.close()

    # colorbar
    fig, ax = plt.subplots(1, 1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm),
                ax=ax, pad=.05, fraction=1, extend='max')
    cbar.set_label('O3 [ppb]')
    ax.axis('off')

    # print info
    print('2.4 million values')
    print('15.0 % missing values in total')
    print('first longer correlated gap')
    print('2011-08-19 09:00:00 -- 2011-08-20 04:00:00 local time')
    print('length 19 h')

    # save colorbar
    save_path = settings.output_dir + 'o3_value_matrix_colorbar.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
    print(f'colorbar saved to {save_path}')
    plt.close()


def mask_matrix():
    """
    the missing, train, test, val masks as a matrix.
    This plot was not published.
    """
    print('mask matrix...')

    # read data
    ug = UBAGraph()
    mask_df = pd.read_csv(ug.mask_path, index_col=0)
    hd = HourlyData('o3')
    hd.read_from_file()
    o3_df = hd.df

    # prepare data
    data_df = pd.DataFrame(index=mask_df.index,
                            columns=['color'])
    for idx, col in enumerate(mask_df.columns):
        data_df[mask_df[col]] = float(idx)
    n_stations = len(o3_df.columns)
    n_timesteps = len(o3_df.index)
    data = np.array(data_df.color.to_list()).reshape((n_stations, n_timesteps))

    # colormap
    cmap = colors.ListedColormap(['white', 'lightgray', 'gray',
                                  'firebrick'])

    # plot
    im = plt.imshow(data, interpolation='none', cmap=cmap)
    ax = plt.gca()
    ax.axis('off')

    # save figure
    save_path = settings.output_dir + 'mask_matrix.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=2000)
    print(f'saved to {save_path}')
    plt.close()


def time_series(gap_id=None):
    """
    A panel visualization of different models on the same time series
    test gap
    """
    print('time series...')

    # gap data
    if not gap_id:
        gap_id = 104526
    ug = UBAGraph()
    gap_df = pd.read_csv(ug.gap_path, index_col=0)
    gap = gap_df.loc[gap_id, :]

    # additional information
    sd = StationData()
    sd.read_from_file()
    print(f'\nmetadata of station {gap.station_id}')
    station_type = sd.df.at[gap.station_id, 'type']
    print(f'station type: {station_type}')
    type_of_area = sd.df.at[gap.station_id, 'type_of_area']
    print(f'coordinates: lon {sd.df.lon[gap.station_id]}' +
          f' lat {sd.df.lat[gap.station_id]}')
    print(f'type of area: {type_of_area}')
    n_neigh, dists = number_of_neighbors(gap.station_id)
    print(f'\n{n_neigh} neighbors of station' +
          f' {gap.station_id} with distances')
    print(f'{dists}\n')

    # range of data to be plotted
    pad = 3
    gap_start_idx = gap.start_idx
    gap_end_idx = gap.start_idx + gap.len
    start_idx = gap_start_idx - pad
    end_idx = gap_end_idx + pad

    # time stamps
    local_time_df = pd.read_csv(ug.reg_path, index_col=0)
    local_time = local_time_df.datetime.to_numpy().reshape(-1)

    # true ozone
    true_o3 = torch.load(ug.y_pt_path).numpy().reshape(-1)

    # modelled ozone
    spatiotemporal_mean_dict = {
        'model_name': 'Spatiotemporal mean',
        'file_without_cs': 'spatiotemporalmean_predictions_useval.pt',
        'file_with_cs': 'SpatiotemporalMean_cs_predictions_useval.pyt',
        'color': 'steelblue'
                     }
    spatial_mean_dict = {
        'model_name': 'Spatial mean',
        'file_without_cs': 'spatialmean_predictions_useval.pt',
        'file_with_cs': 'SpatialMean_cs_predictions_useval.pyt',
        'color': 'steelblue'
                     }
    nearest_neighbor_hybrid_dict = {
        'model_name': 'Nearest neighbor hybrid',
        'file_without_cs': 'nearestneighborhybrid_predictions_useval.pt',
        'file_with_cs': 'NearestNeighborHybrid_cs_predictions_useval.pyt',
        'color': 'steelblue'
                     }
    random_forest_dict = {
        'model_name': 'Random forest',
        'file_without_cs': 'randomforest_predictions_useval.pt',
        'file_with_cs': 'RandomForest_cs_predictions_useval.pyt',
        'color': 'steelblue'
        }
    dicts = [spatiotemporal_mean_dict, spatial_mean_dict,
             nearest_neighbor_hybrid_dict, random_forest_dict]
    model_path = settings.resources_dir + 'models/'
    for dict_ in dicts:
        data_without_cs = torch.load(model_path + dict_['file_without_cs'])
        dict_['data_without_cs'] = data_without_cs.numpy().reshape(-1)
        data_with_cs = torch.load(model_path + dict_['file_with_cs'])
        dict_['data_with_cs'] = data_with_cs.numpy().reshape(-1)

    # setting up plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for model_idx, ax in enumerate(axes):
        dict_ = dicts[model_idx]
        ax.fill_between(range(start_idx, gap_start_idx+1),
                        true_o3[start_idx:gap_start_idx+1],
                        color='gainsboro',
                        alpha=.95,
                        edgecolor=None)
        ax.fill_between(range(gap_start_idx, gap_end_idx+1),
                        true_o3[gap_start_idx:gap_end_idx+1],
                        color='gainsboro',
                        alpha=.5,
                        edgecolor=None)
        ax.fill_between(range(gap_end_idx, end_idx+1),
                        true_o3[gap_end_idx:end_idx+1],
                        color='gainsboro',
                        alpha=.95,
                        edgecolor=None)

        ax.plot(range(gap_start_idx, gap_end_idx+1),
                dict_['data_without_cs'][gap_start_idx:gap_end_idx+1],
                lw=1.3,
                color=dict_['color'],
                linestyle='--',
                alpha=.95,
                label='Without correct and smooth')
        ax.scatter([gap_start_idx, gap_end_idx],
                   [dict_['data_without_cs'][gap_start_idx],
                   dict_['data_without_cs'][gap_end_idx]],
                   s=5,
                   color=dict_['color'])

        ax.plot(range(gap_start_idx, gap_end_idx+1),
                dict_['data_with_cs'][gap_start_idx:gap_end_idx+1],
                lw=1.5,
                color=dict_['color'],
                label='With correct and smooth')
        ax.scatter([gap_start_idx, gap_end_idx],
                   [dict_['data_with_cs'][gap_start_idx],
                   dict_['data_with_cs'][gap_end_idx]],
                   s=3,
                   color=dict_['color'],
                   alpha=.95)

        ax.set_xlim(start_idx, end_idx)
        ax.set_xticks(range(start_idx, end_idx+1))
        ax.set_xticklabels(['']*pad +
                           [local_time[gap_start_idx][5:-6]] +
                           ['']*(gap.len-1) +
                           [local_time[gap_end_idx][5:-6]] +
                           ['']*pad)
        ax.set_yticks(range(0, 65, 10))
        ax.set_ylim(0, 60.)
        ax.legend(loc='upper left', prop={'size':4}, frameon=False,
                  title=dict_['model_name'][0:8])

    # save
    save_path = settings.output_dir + \
                f'time_series_{gap_id}_{n_neigh}_nn.png'
    plt.savefig(save_path, dpi=500,
                bbox_inches='tight', pad_inches=0.1)
    print('saved to' + save_path)


def basic_statistics_and_histogram():
    """
    Histogram and basic statistics of the original data set.
    """
    print('basic statistics and histogram...')

    # read in data
    ug = UBAGraph()
    o3_df = pd.read_csv(ug.y_path, index_col=0)
    o3 = o3_df.values

    # Basic statistics
    n_datapoints = o3.size
    n_missing = np.count_nonzero(np.isnan(o3))
    n_valid = n_datapoints - n_missing
    missing_percentage = n_missing / n_datapoints * 100
    valid_percentage = n_valid / n_datapoints * 100
    mean_o3 = np.nanmean(o3)

    print()
    print(f'n_datapoints: {n_datapoints}')
    print(f'n_valid: {n_valid} (= {valid_percentage:.1f} %)')
    print(f'n_missing: {n_missing} (= {missing_percentage:.1f} %)')
    print(f'mean, std: {np.nanmean(o3):.2f}, {np.nanstd(o3):.2f}')
    print(f'median: {np.nanmedian(o3):.2f}')
    print(f'range: {np.nanmin(o3):.2f}, {np.nanmax(o3):.2f}')
    print(f'10 percentile: {np.nanpercentile(o3, 10):.2f}')
    print(f'25 percentile: {np.nanpercentile(o3, 25):.2f}')
    print(f'50 percentile: {np.nanpercentile(o3, 50):.2f}')
    print(f'75 percentile: {np.nanpercentile(o3, 75):.2f}')
    print(f'90 percentile: {np.nanpercentile(o3, 90):.2f}')
    print(f'95 percentile: {np.nanpercentile(o3, 95):.2f}')
    print(f'99 percentile: {np.nanpercentile(o3, 99):.2f}')

    # Next, plot a histogram and basic statistics
    plt.figure(facecolor='white')
    plt.hist(o3,
             color='gray',
             alpha=.8,
             rwidth=.95, log=False,
             bins=np.arange(0., 160, 5))
    plt.axhline(y=mean_o3, color='red', linestyle='-', linewidth=1, label=f'Mean: {mean_o3:.2f}')  # mean line
    plt.xlabel("$O_3$ (ppbv)")
    plt.ylabel("Frequencey")

    # save
    save_path = settings.output_dir + 'measurement_histogram.png'
    plt.savefig(save_path, dpi=500, facecolor='white')
    print(f'saved to {save_path}')
    plt.close()


def basic_statistics_and_plot():
    """
    Plot and basic statistics of the original data set.
    """
    print('basic statistics and histogram...')

    # read in data
    ug = UBAGraph()
    o3_df1 = pd.read_csv(ug.raw_data_dir + "hourly_o3.csv", index_col=0)
    o3_df2 = pd.read_csv(ug.raw_data_dir + "imputed_o3.csv", index_col=0)
    
    # Convert index to datetime
    o3_df1.index = pd.to_datetime(o3_df1.index)
    o3_df2.index = pd.to_datetime(o3_df2.index)
    
    # Splitting the date components
    o3_df1['month'] = o3_df1.index.month
    o3_df1['day'] = o3_df1.index.day
    
    o3_df2['month'] = o3_df2.index.month
    o3_df2['day'] = o3_df2.index.day
    
    #o3_df1 = o3_df1[o3_df1['month'].isin([9,10,11])]
    #o3_df2 = o3_df2[o3_df2['month'].isin([9,10,11])]

    o3_1 = o3_df1.values.flatten()
    o3_2 = o3_df2.values.flatten()

    # Basic statistics for first dataset
    n_datapoints_1 = o3_1.size
    n_missing_1 = np.count_nonzero(np.isnan(o3_1))
    n_valid_1 = n_datapoints_1 - n_missing_1
    missing_percentage_1 = n_missing_1 / n_datapoints_1 * 100
    valid_percentage_1 = n_valid_1 / n_datapoints_1 * 100
    mean_o3_1 = np.nanmean(o3_1)
    median_o3_1 = np.nanmedian(o3_1)
    
    # Basic statistics for second dataset
    n_datapoints_2 = o3_2.size
    n_missing_2 = np.count_nonzero(np.isnan(o3_2))
    n_valid_2 = n_datapoints_2 - n_missing_2
    missing_percentage_2 = n_missing_2 / n_datapoints_2 * 100
    valid_percentage_2 = n_valid_2 / n_datapoints_2 * 100
    mean_o3_2 = np.nanmean(o3_2)
    median_o3_2 = np.nanmedian(o3_2)

    print()
    print(f'n_datapoints: {n_datapoints_1}')
    print(f'n_valid: {n_valid_1} (= {valid_percentage_1:.1f} %)')
    print(f'n_missing: {n_missing_1} (= {missing_percentage_1:.1f} %)')
    print(f'mean, std: {np.nanmean(o3_1):.2f}, {np.nanstd(o3_1):.2f}')
    print(f'median: {np.nanmedian(o3_1):.2f}')
    print(f'range: {np.nanmin(o3_1):.2f}, {np.nanmax(o3_1):.2f}')
    print(f'10 percentile: {np.nanpercentile(o3_1, 10):.2f}')
    print(f'25 percentile: {np.nanpercentile(o3_1, 25):.2f}')
    print(f'50 percentile: {np.nanpercentile(o3_1, 50):.2f}')
    print(f'75 percentile: {np.nanpercentile(o3_1, 75):.2f}')
    print(f'90 percentile: {np.nanpercentile(o3_1, 90):.2f}')
    print(f'95 percentile: {np.nanpercentile(o3_1, 95):.2f}')
    print(f'99 percentile: {np.nanpercentile(o3_1, 99):.2f}')
    
    print()
    print(f'n_datapoints: {n_datapoints_2}')
    print(f'n_valid: {n_valid_2} (= {valid_percentage_2:.1f} %)')
    print(f'n_missing: {n_missing_2} (= {missing_percentage_2:.1f} %)')
    print(f'mean, std: {np.nanmean(o3_2):.2f}, {np.nanstd(o3_2):.2f}')
    print(f'median: {np.nanmedian(o3_2):.2f}')
    print(f'range: {np.nanmin(o3_2):.2f}, {np.nanmax(o3_2):.2f}')
    print(f'10 percentile: {np.nanpercentile(o3_2, 10):.2f}')
    print(f'25 percentile: {np.nanpercentile(o3_2, 25):.2f}')
    print(f'50 percentile: {np.nanpercentile(o3_2, 50):.2f}')
    print(f'75 percentile: {np.nanpercentile(o3_2, 75):.2f}')
    print(f'90 percentile: {np.nanpercentile(o3_2, 90):.2f}')
    print(f'95 percentile: {np.nanpercentile(o3_2, 95):.2f}')
    print(f'99 percentile: {np.nanpercentile(o3_2, 99):.2f}')

    # Calculate frequency for each concentration bin for both datasets
    bins = np.arange(0., 110, 5)
    hist_1, bin_edges_1 = np.histogram(o3_1, bins=bins)
    hist_2, bin_edges_2 = np.histogram(o3_2, bins=bins)
    bin_centers = (bin_edges_1[:-1] + bin_edges_1[1:]) / 2
    
    # Plot line graph and mean line for both datasets
    plt.figure(facecolor='white')
    plt.plot(bin_centers, hist_1, color='red', alpha=.8, linestyle='-', label='$O_3$ Raw')  # line plot for first dataset
    plt.plot(bin_centers, hist_2, color='blue', alpha=.8, linestyle='-', label='$O_3$ Int ')  # line plot for second dataset
    plt.axvline(x=mean_o3_1, color='red', linestyle='-', linewidth=1, label=f'$O_3$ Raw Mean: {mean_o3_1:.2f} ppbv')  # mean line for first dataset
    plt.axvline(x=median_o3_1, color='red', linestyle='--', linewidth=1, label=f'$O_3$ Raw Median: {median_o3_1:.2f} ppbv')  # mean line for second dataset
    plt.axvline(x=mean_o3_2, color='blue', linestyle='-', linewidth=1, label=f'$O_3$ Int Mean: {mean_o3_2:.2f} ppbv')  # mean line for second dataset
    plt.axvline(x=median_o3_2, color='blue', linestyle=':', linewidth=1, label=f'$O_3$ Int Median: {median_o3_2:.2f} ppbv')  # mean line for second dataset
    
    plt.xlabel("$O_3$ (ppbv)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Automatically adjust layout to prevent squashing
    plt.tight_layout()
    
    # save
    save_path = settings.output_dir + 'measurement_line_plot_comparison.png'
    plt.savefig(save_path, dpi=800, facecolor='white')
    print(f'saved to {save_path}')
    plt.close()
    
    # Plot line graph and mean line for both datasets
    plt.figure(facecolor='white')
    plt.plot(bin_centers, hist_1, color='red', alpha=.8, linestyle='-', label='$O_3$ Raw')  # line plot for first dataset
    plt.axvline(x=mean_o3_1, color='green', linestyle='-', linewidth=1, label=f'$O_3$ Raw Mean: {mean_o3_1:.2f} ppbv')  # mean line for first dataset
    plt.axvline(x=median_o3_1, color='green', linestyle='--', linewidth=1, label=f'$O_3$ Raw Median: {median_o3_1:.2f} ppbv')  # mean line for second dataset
    
    plt.xlabel("$O_3$ (ppbv)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Automatically adjust layout to prevent squashing
    plt.tight_layout()
    
    # save
    save_path = settings.output_dir + 'measurement_line_plot_comparison_raw.png'
    plt.savefig(save_path, dpi=800, facecolor='white')
    print(f'saved to {save_path}')
    plt.close()
    
    # Plot line graph and mean line for both datasets
    plt.figure(facecolor='white')
    plt.plot(bin_centers, hist_2, color='blue', alpha=.8, linestyle='-', label='$O_3$ Int')  # line plot for first dataset
    plt.axvline(x=mean_o3_2, color='green', linestyle='-', linewidth=1, label=f'$O_3$ Int Mean: {mean_o3_2:.2f} ppbv')  # mean line for first dataset
    plt.axvline(x=median_o3_2, color='green', linestyle='--', linewidth=1, label=f'$O_3$ Int Median: {median_o3_2:.2f} ppbv')  # mean line for second dataset
    
    plt.xlabel("$O_3$ (ppbv)")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Automatically adjust layout to prevent squashing
    plt.tight_layout()
    
    # save
    save_path = settings.output_dir + 'measurement_line_plot_comparison_int.png'
    plt.savefig(save_path, dpi=800, facecolor='white')
    print(f'saved to {save_path}')
    plt.close()

def exceedances():
    """
    A bar plot with exceedances before and after the imputation.
    """
    print('exceedances...')

    # get data
    exc_df = count_exceedances()

    # plot style
    plt.style.use('seaborn-darkgrid')
    sns.set(rc={'axes.facecolor':'whitesmoke'})

    # plot
    fig, ax = plt.subplots()
    for idx, row in exc_df[:-4].iterrows():
        ax.bar(row.threshold,
               row.difference,
               4,
               color='steelblue',
               alpha=.8)

    ax.set_xticks(exc_df.threshold[:-4].astype(float))
    ax.set_yscale('log')

    ax.set_xlabel('O3 threshold [ppb]')
    ax.set_ylabel('additional exceedances')

    # save
    save_path = settings.output_dir + 'exceedances.png'
    plt.savefig(save_path, dpi=500)
    print(f'saved to {save_path}')


def true_vs_imputed():
    """
    A heatmap, only isolated gaps, only correlated gaps, summary.
    """
    print('true versus imputed...')
    dicts = [
             {'identifier': '_single',
              'corr_list': [False]}
              ]
    for dict_ in dicts:
        print('\n', dict_['identifier'], '\n')

        # prepare scatter true vs. imputed
        corr_list = dict_['corr_list']
        df = get_characteristics_df()
        y_tru_short = df[(df.test_mask) &
                         (df.gap_len<=2) &
                         (df.correlated.isin(corr_list))].y_true
        y_imp_short = df[(df.test_mask) &
                         (df.gap_len<=2) &
                         (df.correlated.isin(corr_list))].y_imputed
        y_tru_long = df[(df.test_mask) &
                         (df.gap_len>2) &
                         (df.correlated.isin(corr_list))].y_true
        y_imp_long = df[(df.test_mask) &
                         (df.gap_len>2) &
                         (df.correlated.isin(corr_list))].y_imputed
        y_tru = df[df.test_mask].y_true
        y_imp = df[df.test_mask].y_imputed

        # print statistics, only for summary
        for tru, imp, what in [(y_tru_short, y_imp_short, 'short'),
                               (y_tru_long, y_imp_long, 'long'),
                               (y_tru, y_imp, 'summary') ]:
            if (dict_['identifier'] != '') & (what== 'summary'):
                continue
            r2 = r2_score(tru, imp)
            rmse = (mean_squared_error(tru, imp))**.5
            d = index_of_agreement(tru, imp)
            print(what)
            print(f'r2: {r2:.2f}')
            print(f'rmse: {rmse:.2f}')
            print(f'd: {d:.2f}\n')

        # plot style
        plt.style.use('seaborn-darkgrid')
        sns.set(rc={'axes.facecolor':'whitesmoke'})

        # scatter true vs. imputed
        fig, ax = plt.subplots(1, 2)
        min_val = -7.
        max_val = 115

        # Construct 2D histogram from data using the 'plasma' colormap
        norm = LogNorm(vmin=1, vmax=600) # Adjust scale of the heatmap
        colormap = 'OrRd'  # winter, Blues_r, PuBu_r

        ax[0].hist2d(y_tru_short,
                     y_imp_short,
                     bins=(np.arange(min_val, max_val, 1.0),
                           np.arange(min_val, max_val, 1.0)),
                     norm=norm,
                     # cmap="Blues_r"
                     cmap=colormap
                    )

        ax[1].hist2d(y_tru_long,
                     y_imp_long,
                     bins=(np.arange(min_val, max_val, 1.0),
                           np.arange(min_val, max_val, 1.0)),
                     norm=norm,
                     # cmap="Blues_r"
                     cmap=colormap
                    )

        for ax_ in [0, 1]:
            ax[ax_].plot([min_val, max_val],
                         [min_val, max_val],
                         linestyle='--',
                         lw=1.2,
                         color='gray',
                         alpha=.55,
                         zorder=1,
                         )
            ax[ax_].set_aspect('equal', adjustable='box')
            ax[ax_].set_xlim(min_val, max_val)
            ax[ax_].set_ylim(min_val, max_val)
            ax[ax_].set_xlabel('true O3 [ppb]')
            #ax[ax_].set_ylabel('imputed O3 [ppb]')
            ax[ax_].grid(True)
        fig.set_size_inches(10, 5)
        ax[0].set_ylabel('imputed O3 [ppb]')

        # Plot a colorbar with label.
        m = cm.ScalarMappable(cmap=colormap,norm=norm)
        m.set_array([])
        cb = plt.colorbar(m, ax=ax,shrink=0.4)
        cb.set_label('Number of entries')

        # save
        id_ = dict_['identifier']
        true_vs_imp_heat_path = settings.output_dir + \
                                f'true_vs_imp_heatmap{id_}.png'
        plt.savefig(true_vs_imp_heat_path, dpi=500)
        print(f'written to {true_vs_imp_heat_path}')
        plt.close()


def n_neighbors_vs_r2():
    """
    Number of neighbors vs. r2...
    """
    print('number of neighbors vs. r2...')

    # prepare data
    df = get_characteristics_df()
    n_list = np.unique(df.n_neighbors).tolist()
    r2_n_dict = {}
    models = ['y_stm_cs', 'y_sm_cs', 'y_nnh_cs', 'y_rf_cs']
    colors = ['rosybrown', 'lightcoral', 'firebrick', 'blue']
    for model in models:
        r2_n_list = []
        for n in n_list:
            filter_ = [(df.gap_len>2) &
                       (df.test_mask) &
                       (df.n_neighbors==n) &
                       (df.correlated==False)][0]
            df_filtered = df[filter_]
            if len(df_filtered) > 0:
                r2_n_list.append(r2_score(df_filtered.y_true,
                                          df_filtered[model]))
            else:
                r2_n_list.append(np.nan)
        r2_n_dict[model] = r2_n_list

    # plot style
    plt.style.use('seaborn-darkgrid')
    sns.set(rc={'axes.facecolor':'whitesmoke'})

    # line n_neighbors vs. r2
    fig, ax = plt.subplots()
    for color, model in zip(colors, models):
        ax.plot(n_list,
                r2_n_dict[model],
                color=color,
                lw=2.,
                label=model)
        ax.set_ylim(-.1, 1.1)
    ax.set_xlabel('# neighbors')
    ax.set_ylabel('R2 score')
    ax.legend()

    # save
    n_neigh_vs_r2_path = settings.output_dir + 'n_neigh_vs_r2.png'
    plt.savefig(n_neigh_vs_r2_path, dpi=500)
    print(f'written to {n_neigh_vs_r2_path}')
    plt.close()


def gap_len_vs_r2():
    """
    Gap length vs. r2 value
    """
    print('gap length versus r2...')

    # prepare data
    df = get_characteristics_df()
    len_list = list(range(1, 51))
    r2_len_dict = {}
    models = ['y_stm_cs', 'y_sm_cs', 'y_nnh_cs', 'y_rf_cs']
    colors = ['rosybrown', 'lightcoral', 'firebrick', 'blue']
    for model in models:
        r2_len_list = []
        for len_ in len_list:
            filter_ = [(df.gap_len==len_) &
                       (df.test_mask)][0]
            df_filtered = df[filter_]
            if len(df_filtered) > 0:
                r2_len_list.append(r2_score(df_filtered.y_true,
                                            df_filtered[model]))
            else:
                r2_len_list.append(np.nan)
        r2_len_dict[model] = r2_len_list

    # plot style
    plt.style.use('seaborn-darkgrid')
    sns.set(rc={'axes.facecolor':'whitesmoke'})

    # plot
    fig, ax = plt.subplots()
    for color, model in zip(colors, models):
        ax.plot(len_list,
                r2_len_dict[model],
                color=color,
                lw=2.,
                label=model)
    ax.plot([5.5, 5.5],
            [-2, 2],
            color='gray',
            lw=1.5,
            alpha=.5,
            linestyle='--')
    plt.xlim(-.9, 30)
    plt.ylim(0.2, 1.0)
    plt.legend()
    plt.xlabel('gap length')
    plt.ylabel('R2 score')

    # save
    len_vs_r2_path = settings.output_dir + 'len_vs_r2.png'
    plt.savefig(len_vs_r2_path, dpi=500)
    print(f'written to {len_vs_r2_path}')
    plt.close()


def imputed_dataset_matrix():
    """
    Creates a matrix plot of the imputed datset. This plot is used
    for the graphical abstract / TOC figure.
    """
    print('imputed matrix...')

    # read data
    directory = settings.resources_dir + 'imputed_dataset/'
    filename = 'imputed_o3.csv'
    y_imp = pd.read_csv(directory+filename, index_col=0)

    # for reshape and vmax
    hd = HourlyData('o3')
    hd.read_from_file()
    o3_df = hd.df
    n_stations = len(o3_df.columns)
    n_timesteps = len(o3_df.index)
    vmin = 0.
    vmax = np.nanpercentile(o3_df.values, 99)

    # plot
    im = plt.imshow(y_imp.values.T, interpolation='none',
                    vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.axis('off')

    # save figure
    save_path = settings.output_dir + 'imputed_o3_value_matrix.png'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=2000)
    print(f'saved to {save_path}')
    plt.close()


def modeled_time_series():
    """
    A completely modeled time series, as a proof of concept.
    """
    print('modeled time series...')

    # load imputed dataset
    dir_ = settings.resources_dir + 'imputed_dataset/'
    o3_file = 'imputed_o3.csv'
    info_file = 'imputed_info.csv'
    o3_df = pd.read_csv(dir_+o3_file, index_col=0)
    info_df = pd.read_csv(dir_+info_file, index_col=0)

    # load station data
    sd = StationData()
    sd.read_from_file()
    station_df = sd.df

    # find stations without data
    find_stations = False
    if find_stations:
        filter_ = info_df.all(axis=0)
        null_station_list = info_df.columns[filter_].to_list()
        for station in null_station_list:
            print(station)
            print(number_of_neighbors(int(station), return_ids=True))
    chosen_station_ids = [111121, 111123, 111131]

    # plot info on chosen stations
    for station_id in chosen_station_ids:
        n_ngh, _ = number_of_neighbors(station_id)
        print(station_id)
        print(n_ngh, 'neighbors')
        print(sd.df.loc[station_id], '\n')

    # choose time steps to plot
    # datetimes[5086] = 2011-08-01 00:00:00
    # datetimes[5830] = 2011-09-01 00:00:00
    datetimes = o3_df.index.to_list()
    start_idx = 5086
    end_idx = 5830

    # plot
    fig, ax = plt.subplots(3, 1, figsize=(15,7))
    for plt_idx, station_id in enumerate(chosen_station_ids):
        data = o3_df[str(station_id)].to_list()
        _, ngh_dists, ngh_ids = number_of_neighbors(station_id,
                                                    return_ids=True)
        ax[plt_idx].set_facecolor('whitesmoke')
        ax[plt_idx].set_alpha(0.35)
        ax[plt_idx].plot(data[start_idx:end_idx+1],
                         color='steelblue',
                         zorder=2)
        for ngh_id in ngh_ids:
            data = o3_df[str(ngh_id)].to_list()
            ax[plt_idx].plot(data[start_idx:end_idx+1],
                             color='dimgray',
                             alpha=0.5,
                             lw=.4,
                             zorder=1)
        ax[plt_idx].set_xticks([])
        ax[plt_idx].set_xlim(-12, end_idx-start_idx+12)
        ax[plt_idx].set_ylim(-4, 104)
    ax[plt_idx].set_xticks(range(0, end_idx-start_idx+1, 24))
    ax[plt_idx].set_xticklabels([])

    # save figure
    save_pth = settings.output_dir + 'modeled_time_series.png'
    plt.savefig(save_pth, bbox_inches='tight', pad_inches=0.2, dpi=800)
    print(f'saved to {save_pth}')
    plt.close()


if __name__ == '__main__':
    station_loc_on_map_ = True
    station_ids_on_map_ = False
    o3_value_matrix_ = False
    mask_matrix_ = False
    time_series_ = False
    basic_statistics_and_histogram_ = False
    basic_statistics_and_plot_ = False
    exceedances_ = False
    true_vs_imputed_ = False
    n_neighbors_vs_r2_ = False
    gap_len_vs_r2_ = False
    imputed_dataset_matrix_ = False
    modeled_time_series_ = False

    if station_loc_on_map_:
        station_loc_on_map()
    if station_ids_on_map_:
        station_ids_on_map()
    if o3_value_matrix_:
        o3_value_matrix()
    if mask_matrix_:
        mask_matrix()
    if time_series_:
        time_series()
        # for gap_id in range(104526, 104533):
        #    time_series(gap_id)
    if basic_statistics_and_histogram_:
        basic_statistics_and_histogram()
    if basic_statistics_and_plot_:
        basic_statistics_and_plot()
    if exceedances_:
        exceedances()
    if true_vs_imputed_:
        true_vs_imputed()
    if n_neighbors_vs_r2_:
        n_neighbors_vs_r2()
    if gap_len_vs_r2_:
        gap_len_vs_r2()
    if imputed_dataset_matrix_:
        imputed_dataset_matrix()
    if modeled_time_series_:
        modeled_time_series()

