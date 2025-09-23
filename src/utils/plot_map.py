import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_map(data, folder_path, title, file_name, vmin=0, vmax=1, cmap='viridis'):
    lats = np.linspace(-77, 90, 167)
    lons = np.linspace(-180, 180, 360)
    lon_grid, lat_grid = np.meshgrid(lons, lats)


    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    pcm = ax.pcolormesh(lon_grid, lat_grid, data, transform=ccrs.PlateCarree(), shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Set land color to black
    land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='black')
    ax.add_feature(land)

    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label='CO₂ Flux (mol/m²/s)')

    plt.title(title)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    path = folder_path + '/' + timestamp + '_' + file_name + 'plot.png'
    plt.savefig(path, format='png', dpi=300,  bbox_inches='tight')
    plt.close()