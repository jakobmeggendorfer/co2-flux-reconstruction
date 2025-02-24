import pandas as pd

def build_grids(df_month,cell_width=2):
    # Prepare the cells
    nav_lat_grids = get_cell_range(start = -90, end = 90 ,cell_width = cell_width)
    nav_lon_grids = get_cell_range(start = -180, end = 180 ,cell_width = cell_width)
    
    if nav_lat_grids[-1] != 90:
        nav_lat_grids.append(90)
        
    if nav_lon_grids[-1] != 180:
        nav_lon_grids.append(180)
        
    # Build the grids. Store in a list.
    grids_df_lst=[]
    for lat_i in range(len(nav_lat_grids)):
        for lon_j in range(len(nav_lon_grids)):
            if((nav_lat_grids[lat_i] == 90) or (nav_lon_grids[lon_j] == 180)):
                break
            elif ((lat_i == len(nav_lat_grids) - 1) or (lon_j == len(nav_lon_grids) - 1)):
                break
            else:
                _df_ = df_month.loc[
                    (df_month['nav_lat'] >= nav_lat_grids[lat_i]) & 
                    (df_month['nav_lat'] <  nav_lat_grids[lat_i+1]) &
                    (df_month['nav_lon'] >= nav_lon_grids[lon_j]) & 
                    (df_month['nav_lon'] <  nav_lon_grids[lon_j+1])
                                ]
                grids_df_lst.append(_df_)
    
    print(f"\n Total no. of generated cells: {len(grids_df_lst)}")
    
    return grids_df_lst

def get_zoned_df(appended_data):
    '''
    returns multiple dataframes corresponding to 4 different basins
    '''
    
    zone_ARCTIC = appended_data.loc[appended_data['nav_lat'] > 70.0]
    zone_ARCTIC['zone'] = 'ARCTIC'
        
    zone_NORTH_ATLANTIC= appended_data.loc[(appended_data['nav_lon'] >= -75.0) & (appended_data['nav_lon'] <= 0.0)]
    zone_NORTH_ATLANTIC = zone_NORTH_ATLANTIC.loc[(zone_NORTH_ATLANTIC['nav_lat'] >= 10) & (zone_NORTH_ATLANTIC['nav_lat'] <= 70)]
    zone_NORTH_ATLANTIC['zone'] = 'NORTH_ATLANTIC'
    
    zone_EQ= appended_data.loc[(appended_data['nav_lat'] >= -10.0) & (appended_data['nav_lat'] <= 10.0)]
    zone_EQ_PACIFIC_1 = zone_EQ.loc[(zone_EQ['nav_lon'] >= 105.0) & (zone_EQ['nav_lon'] <= 180.0)]
    zone_EQ_PACIFIC_2 = zone_EQ.loc[(zone_EQ['nav_lon'] >= -180.0) & (zone_EQ['nav_lon'] <= -80.0)]
    zone_EQ_PACIFIC = pd.concat([zone_EQ_PACIFIC_1, zone_EQ_PACIFIC_2])
    zone_EQ_PACIFIC['zone'] = 'EQ_PACIFIC'
    
    zone_SOUTHERN_OCEAN = appended_data.loc[appended_data['nav_lat'] <= -45]
    zone_SOUTHERN_OCEAN['zone'] = 'SOUTHERN_OCEAN'
    
    return zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN

def assign_basins(nav_lat, nav_lon):
    '''
    assigns basins to individual latitude and longitude values. Better use as a lambda function.
    '''
    if nav_lat > 70.0:
        return 'ARCTIC'
    elif -75.0 <= nav_lon <= 0.0 and 10 <= nav_lat <= 70: 
        return 'NORTH_ATLANTIC'
    elif -10.0 <= nav_lat <= 10.0:
        if 105.0 <= nav_lon <= 180.0 or -180.0 <= nav_lon <= -80.0:
            return 'EQ_PACIFIC'
    elif nav_lat <= -45:
        return 'SOUTHERN_OCEAN'
    else:
        return 'OTHER'

# data_df['basin_name'] = data_df.apply(lambda row: assign_basins(row['nav_lat'], row['nav_lon']), axis=1)

def get_pure_ocean_df(appended_data):        
    zone_NORTH_ATLANTIC_PATCH= appended_data.loc[(appended_data['nav_lon'] >= -60.0) & (appended_data['nav_lon'] <= -30)]
    zone_NORTH_ATLANTIC_PATCH = zone_NORTH_ATLANTIC_PATCH.loc[(zone_NORTH_ATLANTIC_PATCH['nav_lat'] >= 10) & (zone_NORTH_ATLANTIC_PATCH['nav_lat'] <= 41.7)]
    zone_NORTH_ATLANTIC_PATCH['zone'] = 'NORTH_ATLANTIC'
    
    return zone_NORTH_ATLANTIC_PATCH
