# Ozone-imputation-in-South-Korea
This project makes use of graph machine learning to impute missing ozone data. Publication info follows.

Most statistical models for time series interpolation do not account for the spatial distribution of data. However, the air pollutant concentration data of nearby stations can exhibit strong correlations, while data from distant stations are generally independent. Therefore, graph theory, the foundation of graph machine learning, assigns a weight to represent the influence of spatial distance. This weight is calculated for connections (edges) between 1-h data points (nodes) that fall within the spatial characteristic shared distance.

This approach enhances the estimation accuracy of commonly used statistical models, particularly by accounting for spatiotemporal correlations. In this study, we developed a statistical baseline method (BM) for O3 concentration estimation. This method uses statistical interpolation techniques, including the spatial mean method, spatiotemporal mean method, nearest neighbor hybrid, and random forest, while incorporating spatial distribution weights derived from graph theory.
