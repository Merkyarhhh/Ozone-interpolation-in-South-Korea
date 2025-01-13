"""
This script prepares a pytorch geometric dataset from the AQ-Bench
dataset. It can be used for first graph machine learning tryouts.
"""

# general
import pdb

# data science
import numpy as np
import numpy.random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# pytorch
import torch
from torch.nn import CosineSimilarity
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_networkx

# plotting
import matplotlib.pyplot as plt
import networkx as nx

# own package
import settings
from preprocessing_utils import geo_to_cartesian


class AQBenchGraph(InMemoryDataset):
    """
    The graph version of the AQ-Bench dataset. It serves as a
    toy example for this study.

    Caution, the features are scaled, while the distance between
    nodes is set to be between -.5 and .5!
    """
    def __init__(self):
        super().__init__()

        # load AQ-Bench dataset
        aqb = pd.read_csv(settings.resources_dir + 'aq_bench/' +
                          'AQbench_dataset_FE.csv', index_col=0)
        aqb_info = pd.read_csv(settings.resources_dir + 'aq_bench/' +
                               'AQbench_variables_FE.csv')

        # settings
        self.radius = 100.
        np.random.seed(settings.random_seed)

        # x are the features, scaled or one-hot encoded
        # features = aqb_info[aqb_info['input_target'] ==
        #                              'input']['column_name'].to_list()
        features = ['lat', 'alt', 'relative_alt',
                    'nightlight_5km', 'forest_25km']
        features_info = aqb_info[aqb_info['column_name'].isin(features)]
        x = aqb[features].copy()
        x.lat = np.abs(x.lat)
        for idx, row in features_info.iterrows():
            column_name = row.column_name
            if row.preparation == 'scale':
                scaler = StandardScaler()
                scale_data = x[column_name].to_numpy().reshape(-1, 1)
                x[column_name] = scaler.fit_transform(scale_data)
            if row.preparation == 'one-hot':
                one_hot = pd.get_dummies(x[column_name],
                                         prefix=column_name)
                one_hot = (one_hot - 0.5)
                x = pd.concat([x, one_hot], axis=1)
                del x[column_name]
        if 'lon' in features:
            del x['lon']
        x = torch.tensor(x.values.astype(np.float32))

        # y are the ozone values, unscaled
        y = aqb.o3_average_values.to_numpy(dtype=np.float32)
        y = torch.tensor(y).view(-1, 1)

        # edges (source, target) and edge attributes (weigh).
        x_, y_, z_ = geo_to_cartesian(aqb.lon.values,
                                      aqb.lat.values)
        coords = np.array([x_, y_, z_]).T
        nn = NearestNeighbors()
        cos = CosineSimilarity(dim=0)
        nn.fit(coords)
        _, idx_lists = nn.radius_neighbors(coords,
                                           radius=self.radius)
        src_list, trg_list, wgh_list = [], [], []
        for trg_idx, src_idx_list in enumerate(idx_lists):
            wghs = []
            for src_idx in sorted(src_idx_list):
                src_list.append(src_idx)
                trg_list.append(trg_idx)
                x_1 = x[src_idx, :]
                x_2 = x[trg_idx, :]
                wghs.append(cos(x_1, x_2).clamp_(min=.1).item())
            for wgh in wghs:
                wgh_list.append(wgh / sum(wghs))
        edge_index = np.array([src_list, trg_list])
        edge_index = torch.tensor(edge_index).detach()  # why?
        edge_weight = np.array(wgh_list, dtype=np.float32).reshape(-1, 1)
        edge_weight = torch.tensor(edge_weight)

        # train mask
        n_nodes = len(x)
        train_mask = np.full(n_nodes, True)
        test_indices = np.random.choice(n_nodes,
                                        size=round(.5*n_nodes),
                                        replace=False)
        train_mask[test_indices] = False
        train_mask = torch.tensor(train_mask)

        # position of nodes
        pos = aqb[['lon', 'lat']].to_numpy(dtype=np.float32)
        pos = torch.tensor(pos)

        # create dataset
        data = Data(x=x, edge_index=edge_index,
                    edge_weight=edge_weight, y=y,
                    train_mask=train_mask, pos=pos)
        self.data, self.slices = self.collate([data])


if __name__ == '__main__':
    """
    Create the datasets as specified
    """
    dataset = AQBenchGraph()
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

    # baseline
    x_train = data.x[data.train_mask].numpy()
    y_train = data.y[data.train_mask].numpy().reshape(-1)
    x_test = data.x[~data.train_mask].numpy()
    y_test = data.y[~data.train_mask].numpy().reshape(-1)
    model = RandomForestRegressor(random_state=settings.random_seed)
    # model = LinearRegression()
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)
    rmse = (mean_squared_error(y_test, y_test_hat))**.5
    r2 = r2_score(y_test, y_test_hat)
    print('======================')
    print('Baseline results:')
    print(f'RMSE: {rmse:.2f}, R2: {r2:.2f}')
    ('======================')

    # visualize
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)
    pos = {}
    for idx, lonlat in enumerate(data.pos): pos[idx] = lonlat.numpy()
    # nx.draw_networkx(G, pos=pos, with_labels=False, node_size=15,
    #                  node_color=data.y, width=.2, vmin=15., vmax=40.)
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=.1,
                     node_color=data.y, width=0., vmin=15., vmax=40.,
                     node_shape='8')
    # plt.show()
    plt.savefig(settings.output_dir + 'aqb_graph_overview.png',
                dpi=1000)
