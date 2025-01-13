# general

# data science
import numpy as np
import pandas as pd
import geopandas

# sklearn
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

# own package
import settings
from retrieval_toar_db import StationData
from preprocessing_utils import geo_to_cartesian


extent = [6., 14.8, 47.1, 55.1]
# Projection = ccrs.Orthographic(
#                  central_longitude=(0.5*(extent[0]+extent[1])),
#                  central_latitude=(0.5*(extent[2]+extent[3])))
# crs_proj4 = Projection.proj4_init

# get country borders
df = geopandas.read_file("C:/Users/sjjun/OneDrive/Desktop/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp")
df_de = df.loc[df['ADMIN'] == 'South Korea']
df_de.crs = 'EPSG:4326'
# df_ae = df_de.to_crs(crs_proj4)

# prepare station data
sd = StationData()
sd.read_from_file()
station_df = sd.df

# nearest neighbors
max_dist =8
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
df['neighbors'] = idx_lists
df['distances'] = dist_lists
df.reset_index(inplace=True)

# set up plot
fig, ax = plt.subplots(subplot_kw={'projection': Projection})
ax.set_extent(extent)
ax.add_geometries(
                  # df_ae['geometry'],
                  df_de['geometry'],
                  crs=Projection,
                  # facecolor='gainsboro',
                  facecolor='white',
                  edgecolor='slategray', lw=0.1,
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
           marker='o',
           linewidths=.2,
           cmap='viridis',
           s=20,
           zorder=3,
           edgecolor='black')
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
save_path = settings.output_dir + 'station_loc_on_map.png'
plt.savefig(save_path, dpi=750, bbox_inches='tight', pad_inches=0.)
print(f'plot saved to {save_path}')
plt.close()

