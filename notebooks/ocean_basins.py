import pandas as pd

def get_zoned(appended_data):
    
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
