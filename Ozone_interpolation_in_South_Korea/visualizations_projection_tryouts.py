"""
Projection tryouts for German map.
"""


import geopandas
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io import shapereader

# extent and projection
[lon_min, lon_max, lat_min, lat_max] = [5.8, 15.1, 47.25, 55.1]
# Projection = ccrs.Orthographic(
#         central_longitude=0.5 * (lon_min + lon_max),
#         central_latitude=0.5 * (lat_min + lat_max)
#         )
Projection = ccrs.Orthographic()
# Projection = ccrs.PlateCarree()

# get country borders
resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'
shpfilename = shapereader.natural_earth(resolution, category, name)
df = geopandas.read_file(shpfilename)
poly = df.loc[df['ADMIN'] == 'Germany']['geometry'].values[0]

# plot
ax = plt.axes(projection=Projection)
ax.set_extent([5.8, 15.1, 47.25, 55.1],
              crs=Projection)
ax.add_geometries(poly,
                  crs=Projection,
                  facecolor='gainsboro',
                  edgecolor='slategray',
                  lw=0.1,
                  alpha=.8)

# save plot
save_path = 'germany.png'
plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.)
plt.close()




import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas
from cartopy.io import shapereader

# get country borders
resolution = "10m"
category = "cultural"
name = "admin_0_countries"
shpfilename = shapereader.natural_earth(resolution, category, name)
df = geopandas.read_file(shpfilename)
df_de = df.loc[df["ADMIN"] == "Germany"]


extent = [6., 14.8, 47.1, 55.1]

# plot
crs = ccrs.Orthographic(
    central_longitude=(0.5 * (extent[0] + extent[1])),
    central_latitude=(0.5 * (extent[2] + extent[3])),
)

crs_proj4 = crs.proj4_init

df_de.crs = "EPSG:4326"
df_ae = df_de.to_crs(crs_proj4)

fig, ax = plt.subplots(subplot_kw={"projection": crs})
ax.set_extent(extent)
ax.add_geometries(
    df_ae["geometry"],
    crs=crs,
    facecolor="gainsboro",
    edgecolor="slategray",
    lw=0.1,
    alpha=0.8,
)

# save plot
save_path = "germany.png"
plt.savefig(save_path, dpi=250, bbox_inches="tight", pad_inches=0.0)
plt.close()


