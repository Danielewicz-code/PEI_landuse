"""Turn raw per-tile detections into one clean, non-overlapping field layer.

Pipeline (each step switchable in config `merge:`):
  0. (optional) core ownership filter — drop overlap duplicates up front
  1. R-tree instance merge — union detections that overlap > swallow_threshold
  2. dissolve into connected 'basins'
  3. watershed per basin to split touching fields (parallel)
  4. final shrink / simplify / min area
Steps 1-4 are the 1968 notebook logic, unchanged, with numbers moved to config.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import log


def _main_body(geom):
    from shapely.geometry import MultiPolygon, Polygon
    geom = geom.buffer(0)
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area) if geom.geoms else Polygon()
    return geom


def rtree_merge(polys, min_area, swallow_threshold):
    from rtree import index

    polys = sorted((p for p in polys if p.is_valid and p.area > min_area), key=lambda p: p.area, reverse=True)
    inst, sidx, nid = {}, index.Index(), 0
    for poly in polys:
        merged = False
        for mid in list(sidx.intersection(poly.bounds)):
            ex = inst[mid]
            if poly.intersects(ex):
                inter = poly.intersection(ex).area
                if max(inter / poly.area, inter / ex.area) > swallow_threshold:
                    new = _main_body(ex.union(poly))
                    sidx.delete(mid, ex.bounds)
                    sidx.insert(mid, new.bounds)
                    inst[mid] = new
                    merged = True
                    break
        if not merged and not poly.is_empty:
            inst[nid] = poly
            sidx.insert(nid, poly.bounds)
            nid += 1
    return list(inst.values())


def _watershed_basin(args):
    """Top-level so it can be pickled by ProcessPoolExecutor."""
    import rasterio.features
    import rasterio.transform
    from scipy import ndimage as ndi
    from shapely.geometry import MultiPolygon, shape
    from skimage.segmentation import watershed

    basin, members, px_w, px_h, p = args
    if not basin.is_valid or basin.is_empty:
        return []
    minx, miny, maxx, maxy = basin.bounds
    pad = p["pad"]
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    w = int(np.ceil((maxx - minx) / px_w)); h = int(np.ceil((maxy - miny) / px_h))
    if w <= 0 or h <= 0:
        return []
    tf = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)

    basin_r = np.zeros((h, w), np.uint8)
    rasterio.features.rasterize([(basin, 1)], out=basin_r, transform=tf)

    markers, cid = [], 1
    for poly in members:
        core = poly.buffer(-p["marker_shrink"])
        parts = core.geoms if isinstance(core, MultiPolygon) else [core]
        if core.is_valid and not core.is_empty:
            for c in parts:
                markers.append((c, cid)); cid += 1
        else:
            markers.append((poly, cid)); cid += 1
    if not markers:
        return []
    marker_r = np.zeros((h, w), np.int32)
    rasterio.features.rasterize(markers, out=marker_r, transform=tf)

    labels = watershed(-ndi.distance_transform_edt(basin_r), marker_r, mask=basin_r)

    out = []
    for g, _ in rasterio.features.shapes(labels, mask=labels > 0, transform=tf):
        poly = shape(g)
        if poly.is_valid and poly.area > p["min_area_pre"]:
            s = poly.buffer(-p["final_shrink"]) if p["final_shrink"] else poly
            if s.is_valid and not s.is_empty:
                parts = s.geoms if isinstance(s, MultiPolygon) else [s]
                out.extend(q for q in parts if q.area > p["min_area_final"])
    return out


def merge_detections(cfg, raw_gdf, run_dir, pixel_size):
    """raw_gdf: output of infer.load_raw_detections. pixel_size: (px_w, px_h) in map units."""
    import geopandas as gpd
    from rtree import index
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from tqdm.auto import tqdm

    m = cfg["merge"]
    gdf = raw_gdf
    log(run_dir, f"Raw detections: {len(gdf)}")

    if m.get("core_ownership", False):
        gdf = gdf[gdf["owned"]]
        log(run_dir, f"After core-ownership filter: {len(gdf)}")

    merged = rtree_merge(gdf.geometry.tolist(), m["min_area_rtree"], m["swallow_threshold"])
    log(run_dir, f"After R-tree merge: {len(merged)}")

    if not m.get("watershed", True):
        final = merged
    else:
        union = unary_union(merged)
        basins = [union] if isinstance(union, Polygon) else list(union.geoms)
        sidx = index.Index()
        for i, g in enumerate(merged):
            sidx.insert(i, g.bounds)
        params = {k: m[k] for k in ("pad", "marker_shrink", "final_shrink", "min_area_pre", "min_area_final")}
        tasks = [(b, [merged[i] for i in sidx.intersection(b.bounds) if merged[i].intersects(b)],
                  pixel_size[0], pixel_size[1], params) for b in basins]
        log(run_dir, f"Watershed on {len(tasks)} basins...")
        final = []
        with ProcessPoolExecutor(max_workers=m.get("workers") or os.cpu_count()) as ex:
            for polys in tqdm(ex.map(_watershed_basin, tasks, chunksize=16), total=len(tasks), desc="basins"):
                final.extend(polys)

    out = gpd.GeoDataFrame({"geometry": final}, crs=raw_gdf.crs)
    if m.get("simplify_tolerance"):
        out["geometry"] = out.geometry.simplify(m["simplify_tolerance"], preserve_topology=True)
    out = out[out.is_valid & ~out.is_empty].reset_index(drop=True)
    log(run_dir, f"Final fields: {len(out)}")
    return out
