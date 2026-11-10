"""Tile grid + 'core' ownership. Pure numpy, no GIS deps (easy to test).

Core ownership: with stride S and tile T, each tile 'owns' the central S x S
part of itself (extended out to the image edge for edge tiles). Every pixel of
the image is owned by exactly ONE tile. A detection is kept only by the tile
that owns its centroid -> overlap duplicates disappear without any merging.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    tile_id: str
    row: int        # pixel row of tile top-left
    col: int        # pixel col of tile top-left
    size: int
    # owned (core) region in *tile-local* pixel coordinates, [r0, r1) x [c0, c1)
    core_r0: int
    core_r1: int
    core_c0: int
    core_c1: int

    def owns(self, local_r: float, local_c: float) -> bool:
        return self.core_r0 <= local_r < self.core_r1 and self.core_c0 <= local_c < self.core_c1


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] + tile < length:
        s.append(length - tile)
    return s


def _core_bounds(starts: list[int], i: int, tile: int, length: int) -> tuple[int, int]:
    """Global [lo, hi) owned by tile i along one axis: split at midpoints between
    neighbouring tile centres. Works even for the irregular last tile."""
    centre = starts[i] + tile / 2
    lo = 0 if i == 0 else int(round((starts[i - 1] + tile / 2 + centre) / 2))
    hi = length if i == len(starts) - 1 else int(round((centre + starts[i + 1] + tile / 2) / 2))
    return lo, hi


def make_grid(width: int, height: int, tile: int, overlap: float) -> list[Tile]:
    stride = max(1, int(tile * (1.0 - overlap)))
    rows, cols = _starts(height, tile, stride), _starts(width, tile, stride)
    tiles = []
    for i, r in enumerate(rows):
        g_r0, g_r1 = _core_bounds(rows, i, tile, height)
        for j, c in enumerate(cols):
            g_c0, g_c1 = _core_bounds(cols, j, tile, width)
            tiles.append(Tile(
                tile_id=f"r{i:04d}_c{j:04d}", row=r, col=c, size=tile,
                core_r0=g_r0 - r, core_r1=g_r1 - r, core_c0=g_c0 - c, core_c1=g_c1 - c,
            ))
    return tiles


def group_by_row(tiles: list[Tile]) -> dict[str, list[Tile]]:
    """Tiles grouped per grid row -> one output chunk per row (unit of resumability)."""
    out: dict[str, list[Tile]] = {}
    for t in tiles:
        out.setdefault(t.tile_id.split("_")[0], []).append(t)
    return out
