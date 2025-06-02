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


    x, y = m(lon_grid, lat_grid)
    pcm = m.pcolormesh( x, y, masked_data,  vmin='100', vmax='400', cmap='coolwarm', shading='auto', latlon=True)
    cbar = plt.colorbar(pcm, orientation='vertical', pad=0.01)
    cbar.set_label('Pre-industrial CO2 Fugacity')

    plt.title(f"'Pre-industrial CO2 fugacity")
    plt.show()

def plotMaeInPercent(dataframe):
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

    lon_min, lon_max = -180, 180

    # Create a grid
    num_lat, num_lon = 180, 360  # Grid resolution
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


def plot_feature_on_map(data, lower_lat_bound, upper_lat_bound, vmin, vmax, folder_path, file_name, title="Feature Map"):
    """
    Plots a feature from a 3D numpy array on a Basemap.

    Parameters:
    - data: 3D numpy array of shape (lat, lon, features)
    - lower_lat_bound: float, lower latitude bound of the data
    - upper_lat_bound: float, upper latitude bound of the data
    - feature_index: int, index of the feature to plot
    - title: str, title of the plot (optional)
    - cmap: str, colormap to use (optional)
    """
    # Get latitude and longitude ranges
    n_lat, n_lon = data.shape[0], data.shape[1]
    lat_step = (upper_lat_bound - lower_lat_bound) / n_lat
    lon_step = 360 / n_lon

    lats = np.linspace(lower_lat_bound+lat_step/2, upper_lat_bound-lat_step/2, n_lat)
    lons = np.linspace(-180+lon_step/2, 180-lon_step/2, n_lon)

    # Meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Set up the map
    fig, ax = plt.subplots(figsize=(18, 8))
    m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=ax)
    m.drawcoastlines()

    # Plot with pcolormesh
    cs = m.pcolormesh(lon_grid, lat_grid, data, vmin=vmin, vmax=vmax,shading='auto', cmap='coolwarm', latlon=True)
    m.fillcontinents(color='black')
    # plt.colorbar(cs, label=f"")
    plt.title(title)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    path = folder_path + '/' + timestamp + '_' + file_name + '.png'
    plt.savefig(path, format='png', dpi=300,  bbox_inches='tight') 
    plt.close()