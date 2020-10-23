import pandas as pd
import geopandas as gpd
import descartes
from shapely.geometry import Point
import matplotlib.pyplot as plt

def world_map(df, groups):
    df = df[df['gname']!='Unknown']
    df = df[(df['longitude'] != 0) | (df['latitude'] != 0) | (df['latitude'].notnull()) | (df['longitude'].notnull())]
    df.dropna(subset=['longitude', 'latitude'], inplace=True)
    crs = {'init': 'epsg:4326'}
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    attacks = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[world.continent!='Antarctica']
    attacks = attacks.to_crs(world.crs)
    target_group = attacks[attacks['gname']==group]
    try:
        percentage = str(round((len(target_group)/len(attacks))*100,2)) + '%'
        base = world.plot(color='lightsteelblue', edgecolor='white', alpha=0.4)
        attacks.plot(ax=base, marker='o', color='goldenrod', markersize=0.1, label='Other')
        target_group.plot(ax=base, marker='o', color='darkred', markersize=2, label=group + '   -   (' + percentage + ')')
        lgnd = plt.legend(prop={'size': 10})
        for handle in lgnd.legendHandles:
            handle.set_sizes([50.0])
        plt.show()
    except:
        pass

def map_clusters(df):
    df = df[(df['longitude'] != 0) | (df['latitude'] != 0) | (df['latitude'].notnull()) | (df['longitude'].notnull())]
    df.dropna(subset=['longitude', 'latitude'], inplace=True)
    crs = {'init': 'epsg:4326'}
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    attacks = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[world.continent!='Antarctica']
    base = world.plot(color='lightsteelblue', edgecolor='white', alpha=0.4)
    attacks = attacks.to_crs(world.crs)
    cluster0 = attacks[attacks['cluster']==0]
    cluster1 = attacks[attacks['cluster']==1]
    cluster2 = attacks[attacks['cluster']==2]
    cluster0.plot(ax=base, marker='o', color='goldenrod', markersize=1, label='Cluster0')
    cluster1.plot(ax=base, marker='o', color='darkred', markersize=1, label='Cluster1')
    cluster2.plot(ax=base, marker='o', color='blue', markersize=1, label='Cluster2')
    lgnd = plt.legend(prop={'size': 10})
    for handle in lgnd.legendHandles:
        handle.set_sizes([50.0])
    plt.show()
