"""
This is an old script that was used to visualize data directly from
the TOAR database. Not meant for reuse.
"""

# general
import os
import pdb
import datetime as dt

# data science
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt

# own
import settings
from preprocessing_aqbench import AQBenchGraph
from preprocessing_time_resolved import TimeResolvedOzone


def time_series_lenght():
    """
    Just start and end points of the time series
    """
    # get a data frame with all numids where wind data is available
    query = """
            SELECT station_numid AS id
              FROM parameter_series
              WHERE parameter_name='u'
                AND parameter_measurement_method='model simulation'
              ORDER BY station_numid
              """
    df = query_db(query)
    df.set_index('id', inplace=True)

    # The air pollutants we want to examine
    var_list = ['o3', 'no2', 'no']
    for var in var_list:
        query = f"""
                WITH wind_numid_table AS
                 (SELECT station_numid
                  FROM parameter_series
                  WHERE parameter_name='u'
                    AND parameter_measurement_method='model simulation'
                  ORDER BY station_numid)
                SELECT wind_numid_table.station_numid AS id,
                       data_start_date AS {var}_start_date,
                       data_end_date AS {var}_end_date
                FROM parameter_series
                INNER JOIN wind_numid_table
                ON parameter_series.station_numid=wind_numid_table.station_numid
                WHERE parameter_name='{var}'
                ORDER BY id
                ;
                """
        var_df = query_db(query)
        var_df.set_index('id', inplace=True)

        # This joined df contains at least one entry per id
        var_df = df.join(var_df)

        # Last, plot it!
        plt.figure(figsize=(5, 10))
        counter = 0
        idx_old = -999
        for idx, row in var_df.iterrows():
            # only take a new line if there is a new station id.
            if idx_old != idx:
                counter += 1
            if type(row[f'{var}_start_date']) == pd.Timestamp:
                plt.plot([row[f'{var}_start_date'], row[f'{var}_end_date']],
                         [counter, counter], color='rebeccapurple')
        plt.plot(2*[pd.Timestamp(year=1997, month=1, day=1)],
                 [0, counter], c='tan', zorder=10, lw=.75)
        plt.plot(2*[pd.Timestamp(year=2014, month=12, day=31)],
                 [0, counter], c='tan', zorder=10, lw=.75)
        plt.xlim(pd.Timestamp(year=1970, month=1, day=1),
                 pd.Timestamp(year=2021, month=12, day=31))
        plt.ylim(-10, counter+10)
        plt.title(f'Span of {var} series at cosmo stations, mark 1997-2014',
                  pad=0.)
        plt.yticks([])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.tight_layout(pad=0.3)
        plt.savefig(settings.output_dir+f'series_{var}.pdf')


def missing_values():
    """
    color the time series according to missing values
    """
    print('missing values...')

    # read in data
    tro = TimeResolvedOzone()
    x_df = pd.read_csv(tro.x_path, index_col=0)
    y_df = pd.read_csv(tro.y_path, index_col=0)
    reg_df = pd.read_csv(tro.reg_path, index_col=0)

    print(x_df.columns)

    # reshape to 2d field
    n_stations = len(np.unique(reg_df.station_id))
    n_timesteps = len(np.unique(reg_df.datetime))
    y_2d = y_df.values.reshape(n_stations, n_timesteps)
    print(f'stations: {n_stations}')
    print(f'timesteps: {n_timesteps}')
    print(f'min: {np.nanmin(y_2d)}')
    print(f'max: {np.nanmax(y_2d)}')
    print(f'mean: {np.nanmean(y_2d)}')
    print(f'missing: {np.count_nonzero(np.isnan(y_2d))/(n_stations*n_timesteps)*100}')

    # info
    var = 'o3'

    # plot the data
    z = int(n_timesteps/n_stations)
    plt.figure(figsize=(z*3, 3))
    plt.imshow(y_2d[0:n_stations,0:z*n_stations], interpolation='none')
    plt.yticks([])
    plt.xticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    path = f'{settings.output_dir}missing_{var}.png'
    plt.savefig(path, dpi=250, bbox_inches='tight', pad_inches=0)
    print(f'saved to {path}')
    plt.close()

    bins = np.arange(-5, 120, 5, dtype=int)
    plt.hist(y_df.values, bins=bins, log=True)
    plt.grid()
    path = f'{settings.output_dir}hist_{var}.png'
    plt.savefig(path)
    print(f'saved to {path}')


def visualize_masks():
    """
    Showing the data split.
    """
    # read in data
    tro = TimeResolvedOzone()
    mask_df = pd.read_csv(tro.mask_path, index_col=0)
    reg_df = pd.read_csv(tro.reg_path, index_col=0)

    # prepare data
    n_stations = len(np.unique(reg_df.station_id))
    n_timesteps = len(np.unique(reg_df.datetime))

    missing_o3_mask = mask_df.missing_o3_mask.values.reshape(n_stations, n_timesteps)
    val_mask = mask_df.val_mask.values.reshape(n_stations, n_timesteps)
    test_mask = mask_df.test_mask.values.reshape(n_stations, n_timesteps)

    data = np.zeros((n_stations, n_timesteps))
    data[missing_o3_mask] = 3.
    data[val_mask] = 2.
    data[test_mask] = 1.

    # plot
    z = int(n_timesteps/n_stations)
    plt.figure(figsize=(z*3, 3))
    plt.imshow(data[0:n_stations,0:z*n_stations], interpolation='none',
    cmap='Accent')
    plt.yticks([])
    plt.xticks([])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # save
    path = f'{settings.output_dir}masks.png'
    plt.savefig(path, dpi=250, bbox_inches='tight', pad_inches=0)
    print(f'saved to {path}')
    plt.close()

def visualize_graph():
    """
    Since the edge weights are not functioning in the normal
    networkx...
    """
    dataset = AQBenchGraph()
    data = dataset[0]
    lons = data.pos[:, 0].numpy().reshape(-1)
    lats = data.pos[:, 1].numpy().reshape(-1)
    y = data.y.numpy().reshape(-1)
    edge_index = data.edge_index.numpy()
    num_edges = data.num_edges
    edge_weights = data.edge_weight.numpy().reshape(-1)
    node_edge_colors = ['black' if s==True else 'white' for s in data.train_mask]

    plt.scatter(x=lons, y=lats, c=y, vmin=15., vmax=40.,
                s=35, zorder=10, linewidths=.75,
                edgecolors=node_edge_colors)
    for edge_idx in range(50000):  # range(num_edges):
        node1_idx = edge_index[0, edge_idx]
        node2_idx = edge_index[1, edge_idx]
        node1_lon = lons[node1_idx]
        node1_lat = lats[node1_idx]
        node2_lon = lons[node2_idx]
        node2_lat = lats[node2_idx]
        edge_weight = edge_weights[edge_idx]
        plt.plot([node1_lon, node2_lon]*6,
                 [node1_lat, node2_lat]*6,
                 color='k',
                 lw=.2,
                 alpha=edge_weight)
    plt.show()


if __name__ == '__main__':
    """
    Define what we wish to do, then call the respective functions
    """
    time_series_lenght_ = False
    missing_values_ = True
    visualize_masks_ = False
    visualize_graph_ = False

    if time_series_lenght_: time_series_lenght()
    if missing_values_: missing_values()
    if visualize_masks_: visualize_masks()
    if visualize_graph_: visualize_graph()


"""
print('Incoming node degree vs. error in test set:')
node_degrees = []
absolute_errors = []
for node_idx in range(data.num_nodes):
    if data.train_mask[node_idx].item():
        continue
    edge_weights = data.edge_weight[data.edge_index[1, :]==node_idx]
    edge_weights = torch.sort(edge_weights.view(-1)).values[:-1]
    node_degrees.append(torch.sum(edge_weights).item())
    y = data.y[node_idx].item()
    y_hat = y_soft2[node_idx].item()
    absolute_errors.append(np.abs(y-y_hat))
plt.scatter(node_degrees, absolute_errors, color='navy', alpha=.035)
plt.title('Random Forest on time resolved ozone')
plt.xlabel('incoming node degree')
plt.ylabel('absolute error')
# plt.show()
plt.savefig(settings.output_dir+'cs_time_resolved_node_degree_vs_error.png')

"""
