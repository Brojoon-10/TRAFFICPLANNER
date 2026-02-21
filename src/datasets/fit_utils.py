import lanelet2
import sys
import os
import fiona
from geocube.api.core import make_geocube
from shapely.geometry import Polygon, LineString, mapping
import geopandas as gpd
from matplotlib import pyplot as plt
import rasterio as rio
import cv2 # OpenCV added for drawing lines
import rasterio.features
import numpy as np
from rasterio.mask import mask
import matplotlib as mpl
import pandas as pd

def get_fit_maps(data_folder):
    # This function seems for original nuscenes-like map loading, kept as is.
    # proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.5708374,129.1881128)) Town3
    # proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(35.6475476,128.4026479))
    proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0245645453084227, -73.462179936136))

    filename = data_folder
    laneletmap = lanelet2.io.load(filename, proj)
    # Map name must match map_list in fit_map_env.py

#-----------------------------------HJ ADDED-----------------------------------

    fit_maps = {"Town3" : laneletmap}


    # fit_maps = {"boston_seaport" : laneletmap}

#-----------------------------------HJ ADDED-----------------------------------


    return fit_maps

###############################Line Added###############################

# The function signature is changed to accept two different osm files.
def fit_raster(drivable_area_osm, line_markings_osm, shp_folder):
    # proj is defined within the function, respecting the original structure.
    proj = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0245645453084227, -73.462179936136))

    # --- 1. Drivable Area Rasterization (Original Logic) ---
    data_name = drivable_area_osm # Use the specific OSM for drivable area
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
    # Use a more specific name for the shapefile to avoid confusion.
    shp_name = os.path.join(shp_name,'drivable_area.shp')
    with fiona.open(shp_name, 'w', 'ESRI Shapefile', schema) as c:
        for p in range(len(polys)):
            c.write({
                'geometry':mapping(polys[p]),
                'properties' : {'drivable' : 1}
            })

    shp = gpd.read_file(shp_name)
    raster = make_geocube(
        shp,
        measurements=["drivable"],
        resolution=(0.25,0.25),
    )
    r = raster.drivable.data
    r[np.isnan(r)] = 0
    # Convert to 0 or 1
    drivable_raster_data = (r > 0).astype(np.uint8)
    
    ######################################################
    # --- 2. Lane Markings Rasterization (Added Logic) ---
    ######################################################

    line_map = lanelet2.io.load(line_markings_osm, proj)

    # --- 2.1 Extract Junction polygons from areaLayer ---
    # Junction areas have subtype="Junction" in OSM
    junction_polygons = []
    for area in line_map.areaLayer:
        if 'subtype' in area.attributes and area.attributes['subtype'] == 'Junction':
            # Use outerBoundPolygon() to get the polygon points
            try:
                outer_poly = area.outerBoundPolygon()
                outer_points = [(pt.x, pt.y) for pt in outer_poly]
            except:
                # Fallback: iterate through outer bounds using list()
                outer_points = []
                try:
                    outer_bounds = list(area.outerBound)
                    for outer_bound in outer_bounds:
                        for pt in outer_bound:
                            outer_points.append((pt.x, pt.y))
                except:
                    continue
            if len(outer_points) >= 3:
                junction_poly = Polygon(outer_points)
                if junction_poly.is_valid:
                    junction_polygons.append(junction_poly)
                else:
                    # Try to fix invalid polygon
                    junction_poly = junction_poly.buffer(0)
                    if junction_poly.is_valid:
                        junction_polygons.append(junction_poly)

    print(f"Found {len(junction_polygons)} Junction polygons to exclude lane lines from")

    # Filter out non-boundary linestrings (e.g., centerlines, polygons)
    boundary_ids = set()
    for lane in line_map.laneletLayer:
        boundary_ids.add(lane.leftBound.id)
        boundary_ids.add(lane.rightBound.id)

    excluded_way_ids = set()
    for linestring in line_map.lineStringLayer:
        if linestring.id not in boundary_ids:
            excluded_way_ids.add(linestring.id)

    # Helper function to clip line by removing parts inside junction polygons
    def clip_line_outside_junctions(line_geom, junction_polys):
        """Remove parts of line that are inside any junction polygon."""
        result = line_geom
        for junction_poly in junction_polys:
            if junction_poly.intersects(result):
                result = result.difference(junction_poly)
                if result.is_empty:
                    return None
        return result

    # Helper function to add line(s) to appropriate list
    def add_line_to_list(geom, is_dashed, solid_list, dashed_list):
        """Add LineString or MultiLineString parts to appropriate list."""
        if geom is None or geom.is_empty:
            return
        if geom.geom_type == 'LineString':
            if len(geom.coords) >= 2:
                if is_dashed:
                    dashed_list.append(geom)
                else:
                    solid_list.append(geom)
        elif geom.geom_type == 'MultiLineString':
            for part in geom.geoms:
                if len(part.coords) >= 2:
                    if is_dashed:
                        dashed_list.append(part)
                    else:
                        solid_list.append(part)

    # Classify remaining linestrings into solid and dashed lines
    solid_lines, dashed_lines = [], []
    for linestring in line_map.lineStringLayer:
        if linestring.id in excluded_way_ids:
            continue
        points = [(pt.x, pt.y) for pt in linestring]
        if len(points) < 2: continue
        line_geom = LineString(points)
        is_dashed = 'lane_change' in linestring.attributes and linestring.attributes['lane_change'] == 'yes'

        # Clip line to remove parts inside junction polygons
        clipped_line = clip_line_outside_junctions(line_geom, junction_polygons)
        add_line_to_list(clipped_line, is_dashed, solid_lines, dashed_lines)

    # Use the grid info from the drivable area raster as a reference
    target_transform = raster.rio.transform()
    target_shape = raster.drivable.shape
    
    solid_line_raster = np.zeros(target_shape, dtype=np.uint8)
    dashed_line_raster = np.zeros(target_shape, dtype=np.uint8)
    
    

    # Helper function to convert world coordinates to pixel coordinates
    def world_to_pixel(x, y, transform):
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        return col, row

    # Draw solid lines
    for line in solid_lines:
        for i in range(len(line.coords) - 1):
            p1, p2 = line.coords[i], line.coords[i+1]
            pt1_pixel = world_to_pixel(p1[0], p1[1], target_transform)
            pt2_pixel = world_to_pixel(p2[0], p2[1], target_transform)
            cv2.line(solid_line_raster, pt1_pixel, pt2_pixel, 1, thickness=1)

    # Draw dashed lines
    for line in dashed_lines:
        for i in range(len(line.coords) - 1):
            p1, p2 = line.coords[i], line.coords[i+1]
            pt1_pixel = world_to_pixel(p1[0], p1[1], target_transform)
            pt2_pixel = world_to_pixel(p2[0], p2[1], target_transform)
            cv2.line(dashed_line_raster, pt1_pixel, pt2_pixel, 1, thickness=1)

    # --- 3. Stack all layers and return ---
    # Apply convert_array (flip) to all layers to match the trajectory coordinate system
    final_raster_drivable = convert_array(drivable_raster_data)
    final_raster_solid = convert_array(solid_line_raster)
    final_raster_dashed = convert_array(dashed_line_raster)
    
    # Stack the flipped arrays into a multi-channel raster (C, H, W)
    stacked_raster = np.stack([
        final_raster_drivable,
        final_raster_solid,
        final_raster_dashed
    ], axis=0)
    
    # The original code reshaped to (1, H, W). The calling function will now handle
    # the multi-channel dimension. We return the (C, H, W) array.
    # The original variable `r` is no longer returned directly.
    return stacked_raster

###############################Line Added###############################

def convert_array(arr):
    arr_reversed = arr[::-1]
    # arr_flipped = [row[::-1] for row in arr_reversed]
    return np.array(arr_reversed)

# #--------------------------------For HardCode----------------------------------

# def process_lanegraph_lanelet2(laneletmap, res_meters=1.0, eps=1e-6):
#     """
#     Process Lanelet2 map to create a lane graph structure compatible with NuScenes format.

#     Args:
#         laneletmap: Lanelet2 map object
#         res_meters: resolution for discretizing lanes (meters)
#         eps: epsilon for removing duplicates

#     Returns:
#         dict with:
#             xy: n x 2 (x,y coordinates of all discretized points)
#             in_edges: list of lists (incoming edge indices for each point)
#             out_edges: list of lists (outgoing edge indices for each point)
#             edges: m x 5 (x,y,hcos,hsin,length)
#             edgeixes: m x 2 (v0, v1)
#             ee2ix: dict (v0, v1) -> ei
#     """
#     from collections import defaultdict

#     # Step 1: Extract and discretize centerline for each lanelet
#     lane_graph = {}

#     for lanelet in laneletmap.laneletLayer:
#         # Compute centerline from left and right bounds
#         left_bound = [(pt.x, pt.y) for pt in lanelet.leftBound]
#         right_bound = [(pt.x, pt.y) for pt in lanelet.rightBound]

#         # Interpolate centerline (average of left and right)
#         num_pts = max(len(left_bound), len(right_bound))
#         centerline_pts = []
#         for i in range(num_pts):
#             left_idx = min(i, len(left_bound) - 1)
#             right_idx = min(i, len(right_bound) - 1)
#             center_x = (left_bound[left_idx][0] + right_bound[right_idx][0]) / 2.0
#             center_y = (left_bound[left_idx][1] + right_bound[right_idx][1]) / 2.0
#             centerline_pts.append([center_x, center_y])

#         if len(centerline_pts) < 2:
#             continue

#         # Discretize to uniform resolution
#         discretized = discretize_lane_simple(np.array(centerline_pts), res_meters)
#         discretized = remove_duplicates(discretized, eps)

#         if len(discretized) > 0:
#             lane_graph[lanelet.id] = discretized

#     # Step 2: Build connectivity based on endpoint proximity
#     # This finds which lanelets connect to which (branching and merging)
#     connectivity = defaultdict(lambda: {'outgoing': [], 'incoming': []})

#     for ll1_id, lane1 in lane_graph.items():
#         lane1_end = lane1[-1]  # End point of lane1

#         for ll2_id, lane2 in lane_graph.items():
#             if ll1_id == ll2_id:
#                 continue

#             lane2_start = lane2[0]  # Start point of lane2

#             # Check if lane1 connects to lane2
#             dist = np.linalg.norm(lane1_end - lane2_start)
#             if dist < 1.0:  # Within 1 meter (connection threshold)
#                 connectivity[ll1_id]['outgoing'].append(ll2_id)
#                 connectivity[ll2_id]['incoming'].append(ll1_id)

#     # Step 3: Remove duplicate connections at endpoints to avoid double-counting
#     for intok in list(connectivity.keys()):
#         if intok not in lane_graph:
#             continue
#         for outtok in connectivity[intok]['outgoing']:
#             if outtok not in lane_graph:
#                 continue
#             dist = np.linalg.norm(lane_graph[outtok][0] - lane_graph[intok][-1])
#             if dist <= eps:
#                 # Remove last point of incoming lane to avoid overlap
#                 lane_graph[intok] = lane_graph[intok][:-1]
#                 if len(lane_graph[intok]) < 1:
#                     # Keep at least one point
#                     lane_graph[intok] = lane_graph[intok][:1]

#     # Step 4: Build unified point array and index mapping
#     xys = []
#     laneid2start = {}  # Maps lanelet_id -> starting index in xys array

#     for lid, lane in lane_graph.items():
#         if len(lane) == 0:
#             continue
#         laneid2start[lid] = len(xys)
#         xys.extend(lane.tolist())

#     # Step 5: Build edge connectivity (in_edges and out_edges)
#     in_edges = [[] for _ in range(len(xys))]
#     out_edges = [[] for _ in range(len(xys))]

#     for lid, lane in lane_graph.items():
#         if len(lane) == 0 or lid not in laneid2start:
#             continue

#         start_idx = laneid2start[lid]
#         lane_len = len(lane)

#         # Internal connections within the same lanelet
#         for ix in range(lane_len - 1):
#             point_idx = start_idx + ix
#             next_point_idx = start_idx + ix + 1
#             out_edges[point_idx].append(next_point_idx)
#             in_edges[next_point_idx].append(point_idx)

#         # External connections to other lanelets (branching/merging)
#         last_point_idx = start_idx + lane_len - 1

#         for outtok in connectivity[lid]['outgoing']:
#             if outtok in laneid2start:
#                 next_lanelet_start = laneid2start[outtok]
#                 out_edges[last_point_idx].append(next_lanelet_start)
#                 in_edges[next_lanelet_start].append(last_point_idx)

#     # Step 6: Process edges (compute direction vectors and lengths)
#     edges, edgeixes, ee2ix = process_edges_simple(xys, out_edges, eps)

#     return {
#         'xy': np.array(xys),
#         'in_edges': in_edges,
#         'out_edges': out_edges,
#         'edges': edges,
#         'edgeixes': edgeixes,
#         'ee2ix': ee2ix
#     }

# #--------------------------------For HardCode----------------------------------


def discretize_lane_simple(points, res_meters):
    """
    Discretize a lane to uniform resolution.

    Args:
        points: numpy array of shape (N, 2) with (x, y) coordinates
        res_meters: target resolution in meters

    Returns:
        numpy array of discretized points
    """
    if len(points) < 2:
        return points

    result = [points[0]]
    cumulative_dist = 0.0

    for i in range(1, len(points)):
        segment_vec = points[i] - points[i-1]
        segment_len = np.linalg.norm(segment_vec)

        if segment_len < 1e-6:
            continue

        segment_dir = segment_vec / segment_len

        # Add points along this segment
        remaining_dist = segment_len
        while cumulative_dist + res_meters < segment_len:
            cumulative_dist += res_meters
            new_pt = points[i-1] + cumulative_dist * segment_dir
            result.append(new_pt)
            remaining_dist -= res_meters

        cumulative_dist = res_meters - remaining_dist

        # Add endpoint if it's the last segment
        if i == len(points) - 1:
            result.append(points[i])

    return np.array(result)

def remove_duplicates(points, eps):
    """Remove consecutive duplicate points within epsilon distance."""
    if len(points) == 0:
        return points

    result = [points[0]]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[i-1]) > eps:
            result.append(points[i])

    return np.array(result)

def process_edges_simple(xys, out_edges, eps):
    """
    Process edges from point list and outgoing edge connections.

    Args:
        xys: list of (x, y) coordinates
        out_edges: list of lists containing outgoing edge indices
        eps: minimum edge length threshold

    Returns:
        edges: numpy array of shape (M, 5) with (x, y, hcos, hsin, length)
        edgeixes: numpy array of shape (M, 2) with (start_idx, end_idx)
        ee2ix: dict mapping (start_idx, end_idx) -> edge_index
    """
    edges = []
    edgeixes = []
    ee2ix = {}

    for i in range(len(out_edges)):
        x0, y0 = xys[i]
        for e in out_edges[i]:
            x1, y1 = xys[e]
            diff = np.array([x1 - x0, y1 - y0])
            dist = np.linalg.norm(diff)

            if dist <= eps:
                # Skip very short edges
                continue

            # Normalize direction vector
            diff = diff / dist

            ee2ix[(i, e)] = len(edges)
            edges.append([x0, y0, diff[0], diff[1], dist])
            edgeixes.append([i, e])

    return np.array(edges), np.array(edgeixes), ee2ix