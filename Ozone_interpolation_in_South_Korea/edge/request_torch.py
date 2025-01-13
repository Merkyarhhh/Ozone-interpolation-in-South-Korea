import numpy as np
import pandas as pd
import torch

                                
edge_df = pd.read_csv("/home/tech/tech1/atmos_KU/edge.csv", index_col=0)
src_list = edge_df.source_node_index.to_list()
trg_list = edge_df.target_node_index.to_list()
edge_index = torch.tensor(np.array([src_list, trg_list])).detach()
torch.save(edge_index, "/home/tech/tech1/atmos_KU/edge_index.pt")
del edge_df

weight_df = pd.read_csv("/home/tech/tech1/atmos_KU/weight.csv", index_col=0)
edge_weights = torch.tensor(weight_df.values.astype(np.float32)).view(-1, 1)
torch.save(edge_weights, "/home/tech/tech1/atmos_KU/edge_weights.pt")
del weight_df
