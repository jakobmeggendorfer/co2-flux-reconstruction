import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plotAbsolute(dataframe, feature_to_plot):
   # Define map boundaries
    lat_min, lat_max = -77, 90
    lon_min, lon_max = -180, 180

    # Create a grid
    num_lat, num_lon = 360, 720  # Grid resolution
    lat_grid = np.linspace(lat_min, lat_max, num_lat)
    lon_grid = np.linspace(lon_min, lon_max, num_lon)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    m = Basemap(projection='cyl',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c')
    
    # Initialize grid with zeros
    grid_values = np.zeros_like(lat_grid)

    # Map DataFrame values to the grid
    # Find the closest grid point for each latitude/longitude in the DataFrame
    for _, row in dataframe.iterrows():
        lat_idx = np.argmin(np.abs(lat_grid[:, 0] - row['nav_lat']))
        lon_idx = np.argmin(np.abs(lon_grid[0, :] - row['nav_lon']))
        grid_values[lat_idx, lon_idx] = row[feature_to_plot]

    plt.figure(figsize=(18, 8))
    m.fillcontinents(color='black')

    # Mask the data over land
    masked_data = np.ma.masked_where(grid_values == 0, grid_values)

    m.drawparallels(range(-90, 91, 30), labels=[True, False, False, False], color="lightgrey")
    m.drawmeridians(range(-180, 181, 60), labels=[False, False, False, True], color="lightgrey")

    x, y = m(lon_grid, lat_grid)
    pcm = m.pcolormesh( x, y, masked_data,  vmin='100', vmax='400', cmap='coolwarm', shading='auto', latlon=True)
    cbar = plt.colorbar(pcm, orientation='vertical', pad=0.01)
    cbar.set_label('Pre-industrial CO2 Fugacity')

    plt.title(f"'Pre-industrial CO2 fugacity")
    plt.show()

def plotMaeInPercent(dataframe, feature_to_plot):
   # Define map boundaries
    lat_min, lat_max = -77, 90
    lon_min, lon_max = -180, 180

    # Create a grid
    num_lat, num_lon = 360, 720  # Grid resolution
    lat_grid = np.linspace(lat_min, lat_max, num_lat)
    lon_grid = np.linspace(lon_min, lon_max, num_lon)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    m = Basemap(projection='cyl',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c')
    
    # Initialize grid with zeros
    grid_values = np.zeros_like(lat_grid)

    # Map DataFrame values to the grid
    # Find the closest grid point for each latitude/longitude in the DataFrame
    for _, row in dataframe.iterrows():
        lat_idx = np.argmin(np.abs(lat_grid[:, 0] - row['nav_lat']))
        lon_idx = np.argmin(np.abs(lon_grid[0, :] - row['nav_lon']))
        grid_values[lat_idx, lon_idx] = row[feature_to_plot]

    plt.figure(figsize=(18, 8))
    m.fillcontinents(color='black')

    # Mask the data over land
    masked_data = np.ma.masked_where(grid_values == 0, grid_values)

    m.drawparallels(range(-90, 91, 30), labels=[True, False, False, False], color="lightgrey")
    m.drawmeridians(range(-180, 181, 60), labels=[False, False, False, True], color="lightgrey")

    x, y = m(lon_grid, lat_grid)
    pcm = m.pcolormesh( x, y, masked_data,  vmin='100', vmax='400', cmap='coolwarm', shading='auto', latlon=True)
    cbar = plt.colorbar(pcm, orientation='vertical', pad=0.01)
    cbar.set_label('Pre-industrial CO2 Fugacity')

    plt.title(f"'Pre-industrial CO2 fugacity")
    plt.show()


# data is expected to have three attributes in the third dimension: lat, lon, value
def plotAbsolute_numpy(data, vmin=100, vmax=400, lat_min= -74.5, lat_max=74.5):
    # Define map boundaries

    lon_min, lon_max = -179.5, 179.5

    # Create a grid
    num_lat, num_lon = 150, 360  # Grid resolution
    lat_grid = np.linspace(lat_min, lat_max, num_lat)
    lon_grid = np.linspace(lon_min, lon_max, num_lon)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Initialize Basemap
    m = Basemap(projection='cyl',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180,
                resolution='c')

    # Initialize grid with NaNs (to allow masking)
    grid_values = np.full_like(lat_grid, np.nan, dtype=np.float32)

    # Map data points to the nearest grid index
    lat_idx = np.digitize(data[:, 0], lat_grid[:, 0]) - 1
    lon_idx = np.digitize(data[:, 1], lon_grid[0, :]) - 1

    # Assign values to the grid
    grid_values[lat_idx, lon_idx] = data[:, 3]

    plt.figure(figsize=(18, 8))
    m.fillcontinents(color='black')

    # Plot data on the map
    x, y = m(lon_grid, lat_grid)
    pcm = m.pcolormesh(x, y, grid_values, vmin=vmin, vmax=vmax, cmap='coolwarm', shading='auto')
    m.fillcontinents(color='black')

    # Add colorbar
    cbar = plt.colorbar(pcm, orientation='vertical', pad=0.01)
    cbar.set_label('Pre-industrial CO2 Fugacity')

    plt.title("Pre-industrial CO2 Fugacity")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    path = '../plots/u-net/' + timestamp + '_pae_map.png'
    plt.savefig(path, format='png', dpi=300,  bbox_inches='tight') 
    plt.close()