"""Hedgerow-aware field separation  (merge.method: hedgerow_watershed)

Why the legacy merge leaves merged fields
-----------------------------------------
Legacy step 1 UNIONS any two detections that overlap > 25 %. Two neighbouring
fields whose masks bleed across the hedgerow get unioned into ONE polygon, and
that polygon then becomes ONE watershed marker — so the watershed can never
split them again. On top of that the watershed runs on shape distance only,
so where it does split, the line lands at the geometric middle, not on the
hedgerow.

What this does instead (per raster block, in parallel, resumable)
-----------------------------------------------------------------
1. basin    = union of all detections (where clear land is)
2. markers  = shrunken core of EVERY detection, never unioned. Duplicates of
              the same field (tile overlaps, nested masks) are removed by
              priority: complete, high-score masks win; a marker that ends up
              mostly covered by a better one is dropped.
3. relief   = what the IMAGE says: dark lines (hedgerows/tree rows), light lines
              (roads/lanes), gradient, plus a little shape-distance as tiebreak.
4. watershed on that relief -> boundaries snap to the actual hedgerows.
5. RAG merge: two adjacent segments are joined only if the line between them is
              weak in the image (no hedgerow) — fixes fields cut by tile edges.
6. vectorise, keep the polygons this block owns, shrink by final_shrink to drop
   the hedgerow pixels, drop slivers.

`segment_arrays` is pure numpy/skimage so it can be tuned/tested on its own
(see notebooks/05_tune_merge.ipynb).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import log

DEFAULTS = dict(
    block_core=3072, block_halo=1024, marker_shrink=3.0, min_marker_keep=0.3,
    smooth_sigma=1.5, line_radius_px=6, w_dark=1.0, w_light=0.5, w_grad=0.5,
    w_dist=0.3, dist_scale_px=60, rag_merge=True, rag_threshold=0.30,
)


# ---------------------------------------------------------------------------
# Pure array part
# ---------------------------------------------------------------------------
def _norm(a, valid, pct=99.0):
    v = a[valid]
    hi = np.percentile(v, pct) if v.size else 1.0
    return np.clip(a / max(hi, 1e-6), 0, 1).astype(np.float32)


def line_strength(img: np.ndarray, valid: np.ndarray, p: dict) -> np.ndarray:
    """0..1 map, high on hedgerows / tree rows / roads / sharp edges."""
    import cv2

    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), p["smooth_sigma"])
    r = int(p["line_radius_px"])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dark = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)   # thin dark things: hedgerows
    light = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)    # thin bright things: roads
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    w = (p["w_dark"], p["w_light"], p["w_grad"])
    out = (w[0] * _norm(dark, valid) + w[1] * _norm(light, valid) + w[2] * _norm(grad, valid)) / max(sum(w), 1e-6)
    return out.astype(np.float32)


def _rag_merge(labels: np.ndarray, strength: np.ndarray, thresh: float) -> np.ndarray:
    from skimage.graph import merge_hierarchical, rag_boundary

    if labels.max() < 2:
        return labels
    g = rag_boundary(labels, strength.astype(np.float64), connectivity=2)
    if 0 in g:                                   # background: never merge into it
        for n in list(g.neighbors(0)):
            g[0][n]["weight"] = 1e9

    def weight_boundary(graph, src, dst, n):
        d = {"weight": 0.0, "count": 0}
        a, b = graph[src].get(n, d), graph[dst].get(n, d)
        c = a["count"] + b["count"]
        if n == 0 or src == 0 or dst == 0:
            return {"count": c, "weight": 1e9}
        return {"count": c, "weight": (a["count"] * a["weight"] + b["count"] * b["weight"]) / max(c, 1)}

    def merge_boundary(graph, src, dst):
        pass

    merged = merge_hierarchical(labels, g, thresh=thresh, rag_copy=False, in_place_merge=True,
                                merge_func=merge_boundary, weight_func=weight_boundary)
    merged = merged.astype(np.int32) + 1      # merge_hierarchical relabels from 0
    merged[labels == 0] = 0
    return merged


def segment_arrays(img, basin, markers, p, return_debug=False):
    """img: HxW gray; basin: HxW bool; markers: HxW int32 (0 = none).
    Returns labels HxW int32 (0 = not a field)."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    p = {**DEFAULTS, **p}
    strength = line_strength(img, basin, p)
    dist = ndi.distance_transform_edt(basin)
    relief = (1 - p["w_dist"]) * strength + p["w_dist"] * (1 - np.clip(dist / p["dist_scale_px"], 0, 1))
    markers = np.where(basin, markers, 0)
    labels = watershed(relief, markers, mask=basin).astype(np.int32)
    ws = labels.copy() if return_debug else None
    if p["rag_merge"]:
        labels = _rag_merge(labels, strength, p["rag_threshold"])
    if return_debug:
        return labels, {"strength": strength, "relief": relief, "markers": markers, "watershed": ws}
    return labels


def build_markers(shape, transform, geoms, priority, marker_shrink, px_area, min_keep):
    """Rasterise shrunken cores; best priority burnt last (wins). Drop markers
    that kept < min_keep of their own area (they were duplicates)."""
    import rasterio.features
    from shapely.geometry import MultiPolygon

    order = np.argsort(priority)              # ascending -> best last
    shapes, expected = [], {}
    for lab, i in enumerate(order, start=1):
        core = geoms[i].buffer(-marker_shrink)
        if core.is_empty or not core.is_valid:
            core = geoms[i]
        shapes.append((core, lab))
        expected[lab] = core.area / px_area
    m = np.zeros(shape, np.int32)
    if shapes:
        rasterio.features.rasterize(shapes, out=m, transform=transform)
    counts = np.bincount(m.ravel(), minlength=len(shapes) + 1)
    drop = [lab for lab, e in expected.items() if e > 0 and counts[lab] / e < min_keep]
    if drop:
        m[np.isin(m, drop)] = 0
    return m


# ---------------------------------------------------------------------------
# Block worker (runs in a separate process)
# ---------------------------------------------------------------------------
def _segment_block(task, return_debug=False):
    import rasterio
    import rasterio.features
    from rasterio.windows import Window
    from shapely import wkb
    from shapely.geometry import MultiPolygon, shape
    from shapely.ops import unary_union

    blk, raster_path, geoms_wkb, priority, p = task
    geoms = [wkb.loads(g) for g in geoms_wkb]
    T = blk.size
    with rasterio.open(raster_path) as src:
        W, H = src.width, src.height
        outside = blk.col + T > W or blk.row + T > H
        win = Window(blk.col, blk.row, T, T)
        img = src.read(1, window=win, boundless=outside, fill_value=0 if outside else None)
        tf = src.window_transform(win)
        px_area = abs(tf.a * tf.e)

    valid = ~np.isin(img, p.get("nodata_values", [0]))
    basin = np.zeros(img.shape, np.uint8)
    rasterio.features.rasterize([(g, 1) for g in geoms], out=basin, transform=tf)
    basin = (basin > 0) & valid
    markers = build_markers(img.shape, tf, geoms, np.asarray(priority), p["marker_shrink"], px_area, p["min_marker_keep"])

    res = segment_arrays(img, basin, markers, p, return_debug=return_debug)
    labels, dbg = (res if return_debug else (res, None))

    by_label: dict[int, list] = {}
    for g, v in rasterio.features.shapes(labels, mask=labels > 0, transform=tf):
        by_label.setdefault(int(v), []).append(shape(g))

    out = []
    inv = ~tf
    for v, parts in by_label.items():
        poly = unary_union(parts)
        rp = poly.representative_point()
        col, row = inv * (rp.x, rp.y)
        if not blk.owns(row, col):
            continue
        s = poly.buffer(-p["final_shrink"]) if p["final_shrink"] else poly
        if s.is_empty or not s.is_valid:
            continue
        for q in (s.geoms if isinstance(s, MultiPolygon) else [s]):
            if q.area > p["min_area_final"]:
                out.append(q.wkb)
    if return_debug:
        return out, {"img": img, "basin": basin, "labels": labels, "transform": tf, **dbg}
    return blk.tile_id, out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _params(cfg):
    m = cfg["merge"]
    p = {**DEFAULTS, **(m.get("hedgerow") or {})}
    p.update(final_shrink=m["final_shrink"], min_area_final=m["min_area_final"],
             nodata_values=cfg["inference"].get("nodata_values", [0]))
    return p


def _priority(df):
    """Higher = better marker. Complete masks (not cut by a tile edge) beat
    truncated ones, then by score."""
    if "score" not in df or "touches_edge" not in df:      # old backups: only geometry
        a = df.geometry.area.values
        return a / max(a.max(), 1)
    return (~df["touches_edge"].astype(bool)).astype(float).values * 10 + df["score"].values


def make_tasks(cfg, raw_gdf, raster_path, blocks=None):
    import rasterio
    import rasterio.windows
    from shapely.geometry import box

    from .tiling import make_grid

    p = _params(cfg)
    with rasterio.open(raster_path) as src:
        if blocks is None:
            blocks = make_grid(src.width, src.height, p["block_core"] + 2 * p["block_halo"], stride=p["block_core"])
        sidx = raw_gdf.sindex
        prio = _priority(raw_gdf)
        geoms = raw_gdf.geometry.values
        for b in blocks:
            l, bt, r, t = rasterio.windows.bounds(rasterio.windows.Window(b.col, b.row, b.size, b.size), src.transform)
            idx = sidx.query(box(l, bt, r, t), predicate="intersects")
            if len(idx):
                yield b, str(raster_path), [geoms[i].wkb for i in idx], prio[idx].tolist(), p


def separate_fields(cfg, raw_gdf, run_dir, raster_path):
    import geopandas as gpd
    import pandas as pd
    from shapely import wkb
    from tqdm.auto import tqdm

    out_dir = Path(run_dir) / "segments"
    out_dir.mkdir(exist_ok=True)
    tasks = [t for t in make_tasks(cfg, raw_gdf, raster_path) if not (out_dir / f"{t[0].tile_id}.done").exists()]
    log(run_dir, f"Hedgerow watershed: {len(tasks)} blocks to do")
    workers = cfg["merge"].get("workers") or os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_segment_block, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="blocks"):
            bid, polys = f.result()
            if polys:
                gpd.GeoDataFrame(geometry=[wkb.loads(g) for g in polys], crs=raw_gdf.crs).to_file(
                    out_dir / f"{bid}.gpkg", driver="GPKG")
            (out_dir / f"{bid}.done").touch()

    files = sorted(out_dir.glob("*.gpkg"))
    gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in files], ignore_index=True), crs=raw_gdf.crs) \
        if files else gpd.GeoDataFrame(geometry=[], crs=raw_gdf.crs)
    tol = cfg["merge"].get("simplify_tolerance")
    if tol:
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    log(run_dir, f"Fields after hedgerow watershed: {len(gdf)}")
    return gdf


def debug_area(cfg, raw_gdf, raster_path, x, y, overrides=None):
    """Run ONE block centred on map coordinate (x, y) and return arrays for plotting.
    `overrides` = dict of hedgerow params to try without editing the config."""
    import rasterio

    from .tiling import Tile

    p = _params(cfg)
    if overrides:
        p.update(overrides)
    size = p["block_core"] + 2 * p["block_halo"]
    with rasterio.open(raster_path) as src:
        row, col = src.index(x, y)
    blk = Tile("debug", max(0, row - size // 2), max(0, col - size // 2), size, 0, size, 0, size)
    tasks = list(make_tasks(cfg, raw_gdf, raster_path, blocks=[blk]))
    if not tasks:
        raise ValueError("No detections around that point.")
    t = list(tasks[0]); t[-1] = p
    return _segment_block(tuple(t), return_debug=True)
