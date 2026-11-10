"""Synthetic check of the hedgerow watershed (no GIS libs needed).

Three fields separated by dark hedgerows. The 'detections' bleed across the
hedgerows (basin = everything), field C is cut in two by a tile edge
(two markers, no hedgerow between them). Expect exactly 3 fields, with the
A|B boundary on the hedgerow.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from pei.segment import segment_arrays

rng = np.random.default_rng(0)
H, W = 400, 600
img = np.full((H, W), 170, np.float32) + rng.normal(0, 8, (H, W))
img[:, 290:300] = 60        # hedgerow between A (left) and B/C (right)
img[195:205, 300:] = 60     # hedgerow between B (top right) and C (bottom right)
img = np.clip(img, 1, 254)
basin = np.ones((H, W), bool)

markers = np.zeros((H, W), np.int32)
markers[150:250, 50:200] = 1      # A
markers[50:150, 350:550] = 2      # B
markers[250:350, 320:420] = 3     # C, left fragment (tile 1)
markers[250:350, 470:580] = 4     # C, right fragment (tile 2)


def test_three_fields_boundary_on_hedgerow():
    labels = segment_arrays(img, basin, markers, {})
    ids = [l for l in np.unique(labels) if l]
    assert len(ids) == 3, f"expected 3 fields, got {len(ids)}"
    a, b, c = labels[200, 100], labels[100, 450], labels[300, 360]
    assert len({a, b, c}) == 3
    assert labels[300, 360] == labels[300, 520], "C fragments not re-joined"
    # boundary between A and B/C should sit inside the hedgerow columns
    row = labels[100]
    switch = np.where(row[:-1] != row[1:])[0]
    assert all(285 <= s <= 305 for s in switch), switch


def test_no_rag_keeps_fragments():
    labels = segment_arrays(img, basin, markers, {"rag_merge": False})
    assert len([l for l in np.unique(labels) if l]) == 4


if __name__ == "__main__":
    test_three_fields_boundary_on_hedgerow(); test_no_rag_keeps_fragments(); print("segment OK")
