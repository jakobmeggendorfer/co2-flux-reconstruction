import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def mapDataframeToGridClosest(df, feature_list):
    lower_long_bound = -180
    upper_long_bound = 180
    lower_lat_bound = -75
    upper_lat_bound = 75
    resolution = 0.5
    half_cell_width = resolution / 2
    num_long_cells = int((upper_long_bound - lower_long_bound) / resolution)
    num_lat_cells = int((upper_lat_bound - lower_lat_bound) / resolution)

    # Define grid dimensions
    grid_lon = np.linspace(lower_long_bound+half_cell_width, upper_long_bound-half_cell_width, num_long_cells, endpoint=True)
    grid_lat = np.linspace(lower_lat_bound+half_cell_width, upper_lat_bound-half_cell_width, num_lat_cells, endpoint=True) 

    # Mesh grid creation
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    # Flatten grid coordinates for efficient nearest neighbor search
    grid_points = np.vstack([mesh_lat.ravel(), mesh_lon.ravel()]).T

    # Build KDTree from DataFrame coordinates
    tree = cKDTree(df[['nav_lat', 'nav_lon']].values)

    # Query nearest neighbors
    _, indices = tree.query(grid_points, k=1)


    # Initialize an array to store the mapped values [lat, lon, features]
    grid_data = np.zeros((num_lat_cells, num_long_cells, len(feature_list)))

    # Map dataframe values onto the grid
    for i, col in enumerate(feature_list):
        grid_data[:, :, i] = df[col].values[indices].reshape(num_lat_cells, num_long_cells)

    return grid_data


def mapDataframeToGridAveraged(df, feature_list):
    lower_long_bound = -180
    upper_long_bound = 180
    lower_lat_bound = -75
    upper_lat_bound = 75
    resolution = 1
    num_long_cells = int((upper_long_bound - lower_long_bound) / resolution)
    num_lat_cells = int((upper_lat_bound - lower_lat_bound) / resolution)

    # Add columns indicating which cell each point belongs to
    df = df.copy()
    df['lat_idx'] = ((df['nav_lat'] - lower_lat_bound) // resolution).astype(int)
    df['lon_idx'] = ((df['nav_lon'] - lower_long_bound) // resolution).astype(int)

    # Filter out points outside the defined bounds
    df = df[(df['lat_idx'] >= 0) & (df['lat_idx'] < num_lat_cells) &
            (df['lon_idx'] >= 0) & (df['lon_idx'] < num_long_cells)]

    # Initialize grid with NaNs for empty cells
    grid_data = np.full((num_lat_cells, num_long_cells, len(feature_list)), np.nan)

    # Compute mean for each cell
    for (lat_idx, lon_idx), group in df.groupby(['lat_idx', 'lon_idx']):
        for i, col in enumerate(feature_list):
            grid_data[lat_idx, lon_idx, i] = group[col].mean()

    return grid_data