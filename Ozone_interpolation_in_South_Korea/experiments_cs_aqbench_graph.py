"""
Try the correct and smooth algorithm on AQ-Bench.
"""

# general
import pdb

# data science
import numpy as np

# sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# pytorch
import torch
from models_correct_and_smooth import CorrectAndSmooth

# plotting
import matplotlib.pyplot as plt

# own package
import settings
from preprocessing_aqbench import AQBenchGraph


print('Read in AQBenchGraph')
dataset = AQBenchGraph()
data = dataset[0]

print('Fit baseline model')
x_train = data.x[data.train_mask].numpy()
y_train = data.y[data.train_mask].numpy().reshape(-1)
x_test = data.x[~data.train_mask].numpy()
y_test = data.y[~data.train_mask].numpy().reshape(-1)
model = RandomForestRegressor(random_state=settings.random_seed)
model.fit(x_train, y_train)
y_test_hat = model.predict(x_test)
rmse = (mean_squared_error(y_test, y_test_hat))**.5
r2 = r2_score(y_test, y_test_hat)
print('======================')
print('Baseline results:')
print(f'RMSE: {rmse:.2f}, R2: {r2:.2f}')
('======================')

print('Correct and smooth')
cs = CorrectAndSmooth(num_correction_layers=1000, correction_alpha=.75,
                      num_smoothing_layers=1000, smoothing_alpha=0.4,
                      autoscale=True)  # autoscale is misleading...
x = data.x.numpy()
y_hat = model.predict(x)
y_hat = torch.tensor(y_hat, dtype=torch.float32).view(-1, 1)

y_soft = cs.correct(y_soft=y_hat, y_true=data.y[data.train_mask],
                    mask=data.train_mask, edge_index=data.edge_index,
                    edge_weight=data.edge_weight)
y_test_soft = y_soft[~data.train_mask].numpy()
rmse = (mean_squared_error(y_test, y_test_soft))**.5
r2 = r2_score(y_test, y_test_soft)
print(f'After correct:')
print(f'RMSE: {rmse:.2f}, R2: {r2:.2f}')

y_soft2 = cs.smooth(y_soft=y_soft, y_true=data.y[data.train_mask],
                    mask=data.train_mask, edge_index=data.edge_index,
                    edge_weight=data.edge_weight)
y_test_soft2 = y_soft2[~data.train_mask].numpy()
rmse = (mean_squared_error(y_test, y_test_soft2))**.5
r2 = r2_score(y_test, y_test_soft2)
print(f'After smooth:')
print(f'RMSE: {rmse:.2f}, R2: {r2:.2f}')

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
plt.title('Random Forest on AQ-Bench')
plt.xlabel('incoming node degree')
plt.ylabel('absolute error')
# plt.show()
plt.savefig(settings.output_dir+'cs_aqb_node_degree_vs_error.png')



