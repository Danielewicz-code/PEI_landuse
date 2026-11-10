"""Resumable tiled inference on a huge GeoTIFF.

Output: runs/<id>/tiles/<row>.gpkg — one file per grid row with EVERY raw
detection plus attributes (tile_id, score, owned, touches_edge). Nothing is
thrown away here; the merge stage decides what to keep. If Colab dies, re-run
the same cell: finished rows (.done marker) are skipped.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .config import log
from .preprocess import to_uint8_3ch
from .tiling import Tile, group_by_row, make_grid


def _maybe_local_copy(path: Path, enabled: bool, run_dir) -> Path:
    """Random reads from Drive are slow; copy the raster to Colab's local disk once."""
    if not enabled:
        return path
    local = Path("/content/local_rasters") / path.name
    if not local.exists() or local.stat().st_size != path.stat().st_size:
        local.parent.mkdir(parents=True, exist_ok=True)
        log(run_dir, f"Copying {path.name} to local disk (one-time)...")
        shutil.copy2(path, local)
    return local


def _mask_to_records(binary, score, tile: Tile, transform, img_w, img_h, min_area, raw_mask):
    import rasterio.features
    from shapely.geometry import shape

    ys, xs = np.nonzero(raw_mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    T = tile.size
    # does the (raw) mask touch a tile edge that is NOT the image edge?
    touches = (
        (ys.min() == 0 and tile.row > 0) or (xs.min() == 0 and tile.col > 0)
        or (ys.max() == T - 1 and tile.row + T < img_h) or (xs.max() == T - 1 and tile.col + T < img_w)
    )
    recs = []
    for geom, _ in rasterio.features.shapes(binary, mask=binary == 1, transform=transform):
        poly = shape(geom)
        if poly.is_valid and poly.area > min_area:
            recs.append({"geometry": poly, "tile_id": tile.tile_id, "score": float(score),
                         "owned": bool(tile.owns(cy, cx)), "touches_edge": bool(touches)})
    return recs


def run_inference(cfg, model, run_dir, device, raster_path=None):
    import cv2
    import geopandas as gpd
    import rasterio
    import torch
    from rasterio.windows import Window
    from tqdm.auto import tqdm

    icfg = cfg["inference"]
    raster_path = Path(raster_path or cfg["paths"]["input_raster"])
    raster_path = _maybe_local_copy(raster_path, icfg.get("copy_to_local", False), run_dir)
    out_dir = Path(run_dir) / "tiles"
    out_dir.mkdir(exist_ok=True)

    T = cfg["tiling"]["tile_size"]
    conf_t = icfg["conf_threshold"]
    mask_t = icfg["mask_threshold"]
    k = icfg.get("erosion_kernel", 0)
    kernel = np.ones((k, k), np.uint8) if k else None
    nodata_vals = icfg.get("nodata_values", [0])
    max_nodata = icfg.get("max_nodata_ratio", 0.25)
    min_area = icfg.get("min_polygon_area", 500)
    bs = icfg.get("batch_size", 4)

    with rasterio.open(raster_path) as src:
        W, H, crs = src.width, src.height, src.crs
        tiles = make_grid(W, H, T, cfg["tiling"]["overlap"])
        rows = group_by_row(tiles)
        todo = [r for r in rows if not (out_dir / f"{r}.done").exists()]
        log(run_dir, f"{raster_path.name}: {W}x{H}px, {len(tiles)} tiles in {len(rows)} rows, "
                     f"{len(rows) - len(todo)} rows already done.")

        for row_key in tqdm(todo, desc="rows"):
            records = []
            batch: list[tuple[Tile, "torch.Tensor"]] = []

            def flush():
                if not batch:
                    return
                with torch.no_grad():
                    preds = model([b[1].to(device) for b in batch])
                for (tile, _), p in zip(batch, preds):
                    scores = p["scores"].cpu().numpy()
                    keep = np.where(scores >= conf_t)[0]
                    if not len(keep):
                        continue
                    masks = p["masks"][keep].squeeze(1).cpu().numpy()
                    tf = src.window_transform(Window(tile.col, tile.row, T, T))
                    for m, s in zip(masks, scores[keep]):
                        raw = (m > mask_t).astype(np.uint8)
                        if raw.sum() == 0:
                            continue
                        b = cv2.erode(raw, kernel, iterations=icfg.get("erosion_iterations", 1)) if kernel is not None else raw
                        if b.sum() == 0:
                            continue
                        records.extend(_mask_to_records(b, s, tile, tf, W, H, min_area, raw))
                batch.clear()

            for tile in rows[row_key]:
                outside = tile.col + T > W or tile.row + T > H   # only when raster < tile
                data = src.read(1, window=Window(tile.col, tile.row, T, T),
                                boundless=outside, fill_value=0 if outside else None)
                if np.isin(data, nodata_vals).mean() > max_nodata:
                    continue
                img = torch.from_numpy(to_uint8_3ch(data)).permute(2, 0, 1).float() / 255.0
                batch.append((tile, img))
                if len(batch) >= bs:
                    flush()
            flush()

            if records:
                gpd.GeoDataFrame(records, crs=crs).to_file(out_dir / f"{row_key}.gpkg", driver="GPKG")
            (out_dir / f"{row_key}.done").touch()

    log(run_dir, "Inference finished.")
    return out_dir


def load_raw_detections(run_dir):
    import geopandas as gpd
    import pandas as pd

    files = sorted((Path(run_dir) / "tiles").glob("*.gpkg"))
    if not files:
        raise FileNotFoundError("No tile outputs found — run inference first.")
    parts = [gpd.read_file(f) for f in files]
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
