import lanelet2
import sys
import os
import fiona
from geocube.api.core import make_geocube
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.geometry import box, mapping
import geopandas as gpd
from matplotlib import pyplot as plt
import rasterio as rio
import rasterio.features
import numpy as np
from rasterio.mask import mask
import matplotlib as mpl
import pandas as pd

def get_fit_maps(data_folder):
    # proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.5708374,129.1881128)) Town3
    # proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.6475476,128.4026479))
    proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0245645453084227, -73.462179936136))

    filename = data_folder
    laneletmap = lanelet2.io.load(filename, proj)
    fit_maps = {"Town03" : laneletmap}

    return fit_maps

###############################Origin###############################


def fit_raster(data_folder, shp_folder):
    # proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.5708374,129.1881128))
    # proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.6475476,128.4026479))
    proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0245645453084227, -73.462179936136))

    data_name = data_folder
    lanelet = lanelet2.io.load(data_name, proj)
    max_y = 0
    poly = lanelet
    bounds = []
    polys = []
    for lane in lanelet.laneletLayer:
        ll = lane.leftBound
        lr = lane.rightBound
        pts_ll = []
        pts_lr = []
        for pts in ll:
            if pts.y > max_y:
                max_y = pts.y
            pt = (pts.x, pts.y)
            pts_ll.append(pt)
        for pts in lr:
            if pts.y > max_y:
                max_y = pts.y
            pt = (pts.x, pts.y)
            pts_lr.insert(0,pt)
        points = pts_ll + pts_lr
        p = Polygon(points)
        bounds.append(p.bounds)
        polys.append(p)
        
    min_x = min(bounds)[0]
    min_y = min(bounds)[1]
    max_x = max(bounds)[2]

    schema = {
        'geometry': 'Polygon',
        'properties' : {'drivable': 'int'},
    }
    shp_name = shp_folder
    shp_name = os.path.join(shp_name,'test.shp')
    with fiona.open(shp_name, 'w', 'ESRI Shapefile', schema) as c:
        for p in range(len(polys)):
            c.write({
                'geometry':mapping(polys[p]),
                'properties' : {'drivable' : 1}
            })

    shp = gpd.read_file(shp_name)
    # print(shp)
    raster = make_geocube(
        shp,
        measurements=["drivable"],
        resolution=(0.25,0.25),
    )
    r = raster.drivable.data
    # print(raster)
    r[np.isnan(r)] = 0
    # fig,axs = plt.subplots(2,1)
    # axs[0].matshow(r)
    r = convert_array(r)
    # print(np.sum(r,axis=0))
    # axs[1].matshow(r)
    r = r.reshape((1,r.shape[0],r.shape[1]))
    # cmap = mpl.colors.ListedColormap(['black'])
    # raster.drivable.plot()
    # plt.show()

    return r

###############################Origin###############################

def convert_array(arr):
    arr_reversed = arr[::-1]
    # arr_flipped = [row[::-1] for row in arr_reversed]
    return np.array(arr_reversed)
