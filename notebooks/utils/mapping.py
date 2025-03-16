import numpy as np
from scipy.spatial import cKDTree

def mapDataframeToGrid(df, feature_list):
    # Define grid dimensions
    grid_lon = np.linspace(-179.5, 179.5, 360, endpoint=True)   # 360 points (1° resolution)
    grid_lat = np.linspace(-74.5, 74.5, 150, endpoint=True)    # 150 points (1° resolution)

    # Mesh grid creation
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    # Flatten grid coordinates for efficient nearest neighbor search
    grid_points = np.vstack([mesh_lat.ravel(), mesh_lon.ravel()]).T

    # Build KDTree from DataFrame coordinates
    tree = cKDTree(df[['nav_lat', 'nav_lon']].values)

    # Query nearest neighbors
    _, indices = tree.query(grid_points, k=1)


    # Initialize an array to store the mapped values [lat, lon, features]
    grid_data = np.zeros((150, 360, len(feature_list)))

    # Map dataframe values onto the grid
    for i, col in enumerate(feature_list):
        grid_data[:, :, i] = df[col].values[indices].reshape(150, 360)

    return grid_data