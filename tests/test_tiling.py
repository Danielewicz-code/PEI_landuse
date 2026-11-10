"""Run: python -m pytest tests/  (or just python tests/test_tiling.py)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pei.tiling import make_grid


def check(width, height, tile, overlap):
    owners = np.zeros((height, width), np.int32)
    for t in make_grid(width, height, tile, overlap):
        r0, r1 = t.row + t.core_r0, t.row + t.core_r1
        c0, c1 = t.col + t.core_c0, t.col + t.core_c1
        assert 0 <= t.core_r0 < t.core_r1 <= tile and 0 <= t.core_c0 < t.core_c1 <= tile, t
        owners[r0:r1, c0:c1] += 1
    assert (owners == 1).all(), f"{width}x{height} T{tile} ov{overlap}: min {owners.min()} max {owners.max()}"


def test_every_pixel_owned_once():
    for w, h in [(5000, 3000), (1024, 1024), (1500, 1100), (4097, 2049), (800, 600)]:
        for ov in (0.0, 0.25, 0.5):
            check(w, h, 1024, ov)


if __name__ == "__main__":
    test_every_pixel_owned_once(); print("tiling OK")
