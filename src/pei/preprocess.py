"""Image normalisation + preparing huge rasters (grayscale, resample, histogram match).

rasterio is imported lazily so the training code doesn't need it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Per-tile normalisation (used by BOTH training and inference — must not differ)
# ---------------------------------------------------------------------------
def to_uint8_3ch(img: np.ndarray) -> np.ndarray:
    """HxW / HxWx1 / HxWx3 / CxHxW -> HxWx3 uint8 (same rules as the 1968 notebook)."""
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.moveaxis(img, 0, -1)  # CHW -> HWC (rasterio order)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    if np.issubdtype(img.dtype, np.floating):
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.dtype != np.uint8 and img.max() > 255:
        raise ValueError(
            f"Image is {img.dtype} with values > 255. Run pei.preprocess.prepare_raster "
            "first so the model sees the same 0-255 grayscale it was trained on."
        )
    return img.astype(np.uint8)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """(3,H,W) or (H,W,3) -> (H,W) float32, ITU-R 601 luma (what 'grayscale' usually means)."""
    if rgb.shape[0] == 3 and rgb.ndim == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    else:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r.astype(np.float32) + 0.587 * g + 0.114 * b


# ---------------------------------------------------------------------------
# Histogram matching (make 2020 grayscale "look like" 1968 film)
# ---------------------------------------------------------------------------
def sample_histogram(path, n_windows=200, win=512, band_mode="gray", seed=0, nodata_values=(0, 255)):
    """256-bin histogram from random windows of a huge raster (never loads it all)."""
    import rasterio
    from rasterio.windows import Window

    rng = np.random.default_rng(seed)
    hist = np.zeros(256, np.float64)
    with rasterio.open(path) as src:
        for _ in range(n_windows):
            c = int(rng.integers(0, max(1, src.width - win)))
            r = int(rng.integers(0, max(1, src.height - win)))
            w = Window(c, r, min(win, src.width), min(win, src.height))
            if band_mode == "gray" and src.count >= 3:
                a = rgb_to_gray(src.read([1, 2, 3], window=w))
            else:
                a = src.read(1, window=w).astype(np.float32)
            a = np.clip(a, 0, 255).astype(np.uint8)
            valid = ~np.isin(a, nodata_values)
            hist += np.bincount(a[valid].ravel(), minlength=256)
    return hist


def matching_lut(src_hist, ref_hist) -> np.ndarray:
    """LUT (256,) mapping source grey levels -> reference grey levels via CDF matching."""
    s = np.cumsum(src_hist); s /= s[-1]
    r = np.cumsum(ref_hist); r /= r[-1]
    return np.searchsorted(r, s - 1e-12, side="left").clip(1, 254).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full-raster preparation: RGB -> gray -> (resample) -> (hist match) -> tiled GeoTIFF
# ---------------------------------------------------------------------------
def prepare_raster(src_path, dst_path, *, target_res=None, reference_path=None,
                   block=2048, nodata_out=0, log=print):
    """Stream a huge raster to a model-ready single-band uint8 tiled GeoTIFF.

    target_res: metres/pixel to resample to (match the 1968 GSD). None = keep.
    reference_path: a 1968 raster to histogram-match against. None = skip.
    Values 0 are reserved for nodata (real pixels are clipped to 1..254) so the
    'black border' tile filter keeps working.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    src_path, dst_path = Path(src_path), Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    lut = None
    if reference_path:
        log("Sampling histograms for matching...")
        lut = matching_lut(sample_histogram(src_path), sample_histogram(reference_path))

    with rasterio.open(src_path) as src:
        vrt_kw = {}
        if target_res:
            l, b, r, t = src.bounds
            w = int(round((r - l) / target_res)); h = int(round((t - b) / target_res))
            from rasterio.transform import from_origin
            vrt_kw = dict(transform=from_origin(l, t, target_res, target_res), width=w, height=h,
                          resampling=Resampling.average)
        with WarpedVRT(src, **vrt_kw) as vrt:
            profile = dict(driver="GTiff", dtype="uint8", count=1, width=vrt.width, height=vrt.height,
                           crs=vrt.crs, transform=vrt.transform, nodata=nodata_out,
                           tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                           predictor=2, BIGTIFF="YES")
            log(f"Writing {vrt.width}x{vrt.height} -> {dst_path}")
            with rasterio.open(dst_path, "w", **profile) as dst:
                for row in range(0, vrt.height, block):
                    for col in range(0, vrt.width, block):
                        w = Window(col, row, min(block, vrt.width - col), min(block, vrt.height - row))
                        if vrt.count >= 3:
                            data = vrt.read([1, 2, 3], window=w)
                            empty = (data == 0).all(axis=0)
                            g = rgb_to_gray(data)
                        else:
                            g = vrt.read(1, window=w).astype(np.float32)
                            empty = g == 0
                        g = np.clip(g, 1, 254).astype(np.uint8)
                        if lut is not None:
                            g = lut[g]
                        g[empty] = nodata_out
                        dst.write(g, 1, window=w)
            dst_overviews = [2, 4, 8, 16, 32]
            with rasterio.open(dst_path, "r+") as dst:
                dst.build_overviews(dst_overviews, Resampling.average)
    log("Done.")
    return dst_path
