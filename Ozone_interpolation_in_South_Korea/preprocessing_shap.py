# import packages 
import pandas as pd 
import numpy as np 
import xgboost
from xgboost import XGBRegressor, plot_importance 
from sklearn.model_selection import train_test_split
import shap 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

file_path = "C:/Users/ATMOS/Desktop/data/raw_city.csv"
raw_data = pd.read_csv(file_path)

raw_data = raw_data.dropna(axis=0,inplace = False, subset=["Hour_of_day", "Day_of_week", "Day_of_year", "O3", "NO2", "CO", "SO2", "PM10", "PM25", "Temperature", "Wind_speed", "Relative_humidity"])
raw_data = raw_data[["Hour_of_day", "Day_of_week", "Day_of_year", "O3", "NO2", "CO", "SO2", "PM10", "PM25", "Temperature", "Wind_speed", "Relative_humidity"]]

# Features and target
features = ["Hour_of_day", "Day_of_week", "Day_of_year", "NO2", "CO", "SO2", "PM10", "PM25", "Temperature", "Wind_speed", "Relative_humidity"]
target = ["O3"]

# Prepare data for training
X = raw_data[features]
y = raw_data[target]

# load data 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

#---------------- 1년 중 특정 행을 뽑아서 test set 설정하는 코드
# Define the cycle and test ranges
cycle_length = 8760
test_ranges = [(1, 744), (3625, 4344)]

# Initialize lists to hold test and train indices
test_indices = []
train_indices = []

# Calculate the number of cycles in the dataset
num_cycles = len(raw_data) // cycle_length

# Gather indices for each cycle
for cycle in range(num_cycles):
    # Shift indices for each cycle
    offset = cycle * cycle_length
    for start, end in test_ranges:
        test_indices.extend(range(offset + start - 1, offset + end))  # Adjusting for Python's 0-indexing

# All other indices are for training
train_indices = [i for i in range(len(raw_data)) if i not in test_indices]

# Split into train and test DataFrames
test_df = raw_data.iloc[test_indices]
train_df = raw_data.iloc[train_indices]

#-------------

X_train = train_df[features]
X_test = test_df[features]
y_train = train_df[target]
y_test = test_df[target]

# modeling2
model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
r2 = r2_score(y_train, y_pred)
rmse = (mean_squared_error(y_train, y_pred)) ** 0.5

y_train2 = np.array(y_train)
y_pred2 = np.array(y_pred)
a = np.sum((y_train-y_pred)**2)
b = np.sum((np.abs(y_pred2-np.mean(y_train2)) + np.abs(y_train2-np.mean(y_train2)))**2)
IOA = 1 - a/b

print(r2)
print(rmse)
print(IOA)

# load js 
shap.initjs()
explainer = shap.Explainer(model)
shap_values = explainer(X_train)

order = ["Hour_of_day", "Day_of_week", "Day_of_year", "NO2", "CO", "SO2", "PM10", "PM25", "Temperature", "Wind_speed", "Relative_humidity"]
col2num = {col: i for i, col in enumerate(X.columns)}
order = list(map(col2num.get, order))

# 전체 데이터에 대한 SHAP 시각화 
# plot
#shap.plots.force(shap_values, show=False)
shap.plots.beeswarm(shap_values, max_display=15, order=order)
shap.plots.bar(shap_values, max_display=15, order=order)
shap.plots.waterfall(shap_values[0], max_display=15)
shap.plots.bar(shap_values[0], max_display=15, order=order)
shap.plots.waterfall(shap_values[1], max_display=15)
shap.plots.bar(shap_values[1], max_display=15, order=order)
#shap.plots.scatter(shap_values[:, "NO2"], color=shap_values)

# main effect on the diagonal
# specific effect off the diagonal
#shap_raw_values = explainer.shap_values(X_train)
# dependence_plot
#top_inds = np.argsort(-np.sum(np.abs(shap_raw_values), 0))  # (13, ) : 각각의 Feature 에 대해 shap value 다 더한 것 
# make SHAP plots of the three most important features
#for i in range(12):
    #shap.dependence_plot(top_inds[i], shap_raw_values, X_train)

# interact effect off the diagonal
#shap_interaction_values = shap.Explainer(model).shap_interaction_values(X_train)
# interaction_plot
#shap.summary_plot(shap_interaction_values, X_train, max_display=15)
#shap.dependence_plot(("O3", "NO2"), shap_interaction_values, X_train, display_features=X_train)
#shap.dependence_plot(("O3", "NO"), shap_interaction_values, X_train, display_features=X_train)
#shap.dependence_plot(("O3", "CO"), shap_interaction_values, X_train, display_features=X_train)
#shap.dependence_plot(("O3", "PM25"), shap_interaction_values, X_train, display_features=X_train)

