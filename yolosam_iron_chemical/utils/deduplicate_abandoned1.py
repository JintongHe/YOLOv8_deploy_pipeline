import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, box, MultiPolygon
import numpy as np
from tqdm import tqdm
import time
import concurrent.futures
import threading


# def _deduplicate(processed_indices, lock, row1, row2, CLSES, CONFS, poly1, poly2, overlap_percent, method):
#     if CLSES[row1] != CLSES[row2]:
#         pass
#     if row1 == row2 or row1 in processed_indices or row2 in processed_indices:
#         pass
#     area1 = poly1.area
#     area2 = poly2.area
#     over_area = poly1.intersection(poly2).area
#     union_area = poly1.union(poly2).area
#     Ratio = over_area / union_area
#
#     if over_area/area1 > overlap_percent or over_area/area2 > overlap_percent:
#         if method == "area":
#             if area1 >= area2:
#                 if row1 not in processed_indices:
#                     processed_indices.append(row1)
#             else:
#                 if row2 not in processed_indices:
#                     processed_indices.append(row2)
#         elif method == "conf":
#             if CONFS[row1] >= CONFS[row2]:
#                 if row2 not in processed_indices:
#                     processed_indices.append(row2)
#             else:
#                 if row1 not in processed_indices:
#                     processed_indices.append(row1)
#         elif method == "wght":
#             if CONFS[row1] * 0.6 + 0.4 * Ratio >= CONFS[row2] * 0.6 + 0.4 * (1-Ratio):
#                 if row2 not in processed_indices:
#                     processed_indices.append(row2)
#             else:
#                 if row1 not in processed_indices:
#                     processed_indices.append(row1)


def _deduplicate(patch):
    it = tqdm(patch.iterrows(), total=patch.shape[0])
    for idx, Row in it:
        row1 = idx
        row2 = Row["index_right"]
        cls1 = int(Row["clses_left"])
        cls2 = int(Row["clses_right"])
        conf1 = float(Row["confs_left"])
        conf2 = float(Row["confs_right"])

        if cls1 != cls2:
            continue
        if row1 == row2 or row1 in processed_indices or row2 in processed_indices:
            continue

        poly1 = gdf.at[row1, 'geometry']
        poly2 = gdf.at[row2, 'geometry']
        area1 = poly1.area
        area2 = poly2.area
        over_area = poly1.intersection(poly2).area

        union_area = poly1.union(poly2).area
        Ratio = over_area / union_area

        if over_area/area1 > overlap_percent or over_area/area2 > overlap_percent:
            if method == "area":
                if area1 >= area2:
                    if row1 not in processed_indices:
                        processed_indices.append(row1)
                else:
                    if row2 not in processed_indices:
                        processed_indices.append(row2)
            elif method == "conf":
                if conf1 >= conf2:
                    if row2 not in processed_indices:
                        processed_indices.append(row2)
                else:
                    if row1 not in processed_indices:
                        processed_indices.append(row1)
            elif method == "wght":
                if conf1 * 0.6 + 0.4 * Ratio >= conf2 * 0.6 + 0.4 * (1-Ratio):
                    if row2 not in processed_indices:
                        processed_indices.append(row2)
                else:
                    if row1 not in processed_indices:
                        processed_indices.append(row1)


def deduplicate(cfg, method="conf"):
    processes = 8

    boxes = []
    confs = []
    clses = []
    masks = []

    lock = threading.Lock()
    processed_indices = list()

    result = cfg["result"]
    overlap_percent = 0.2
    deduplicate_mode = cfg["deduplicate_mode"]

    if deduplicate_mode == "box":
        geometry = [box(p[0], p[1], p[2], p[3]) for p in result["boxes"]]
        gdf = gpd.GeoDataFrame(
            {
                'geometry': geometry,
                'confs': result["confs"],
                'clses': result["clses"]
            })

    elif deduplicate_mode == "mask":
        geometry = [Polygon(p).buffer(0.00001) for p in result["masks"]]
        boxes_ = [box(p[0], p[1], p[2], p[3]) for p in result["boxes"]]
        gdf = gpd.GeoDataFrame(
            {
                'geometry': geometry,
                'boxes': boxes_,
                'confs': result["confs"],
                'clses': result["clses"]
            })

    joined_gdf = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')

    length = joined_gdf.shape[0]
    interval = length // processes

    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        for i in range(length):
            start = i * interval
            end = (i + 1) * interval if i != processes - 1 else length
            patch = joined_gdf[start:end]
            executor.map(_deduplicate, patch)

        # for idx, Row in tqdm(joined_gdf.iterrows(), total=joined_gdf.shape[0]):
        #     row1 = idx
        #     row2 = Row['index_right']
        #     poly1 = gdf.at[row1, 'geometry']
        #     poly2 = gdf.at[row2, 'geometry']
        #     executor.submit(_deduplicate, processed_indices, lock, row1, row2, CLSES, CONFS, poly1, poly2, overlap_percent, method)


    gdf = gdf.drop(index=list(processed_indices))

    if deduplicate_mode == "box":
        boxes_gdf = gdf['geometry'].tolist()
    elif deduplicate_mode == "mask":
        masks_gdf = gdf["geometry"].tolist()
        boxes_gdf = gdf["boxes"].tolist()
    confs_gdf = gdf["confs"].tolist()
    clses_gdf = gdf["clses"].tolist()

    length = len(boxes_gdf)

    for t in range(length):
        if deduplicate_mode == "box":
            boxes.append(boxes_gdf[t].bounds)
        elif deduplicate_mode == "mask":
            tmpmask = masks_gdf[t]
            if isinstance(tmpmask, MultiPolygon):
                masks_gdf[t] = max(masks_gdf[t].geoms, key=lambda p: p.area)
            masks.append(masks_gdf[t].exterior.coords)
            boxes.append(boxes_gdf[t].bounds)
        confs.append(confs_gdf[t])
        clses.append(clses_gdf[t])

    boxes = np.array(boxes).astype(np.int32)
    confs = np.array(confs)
    clses = np.array(clses).astype(np.int32)
    masks = [np.array(item).astype(np.int32) for item in masks] if deduplicate_mode == "mask" else None
    cfg["result_after_deduplicate"] = dict(boxes=boxes, confs=confs, clses=clses, masks=masks)
    assert len(boxes) == len(confs) == len(clses)==len(masks), "{} vs {} vs {}".format(len(boxes), len(confs), len(clses), len(masks))
    return cfg




