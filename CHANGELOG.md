# Changelog

One line per change that affects results: what changed, why, and which runs are affected.
Newest first.

## 2026-11-10 — v0.1.0: notebook → package
- Moved `clearland.ipynb` logic into `src/pei/` with configs. **The default config reproduces the 1968 behaviour.**
- Fixed: `mask_threshold` was 0.50 in the settings but 0.65 was hard-coded in the loop. The config now says 0.65, which is what was actually used.
- Fixed: the nodata check tested `== 256`, which a uint8 can never equal. It's now `nodata_values: [0]`, the same effective behaviour, and configurable.
- Fixed: train/val split was random on every run with no seed, so fine-tune validation could leak. It's now frozen in `split.json`.
- Changed: final `min_area_final` (1000 m²) now applies to every field. Before, it only applied to fields that broke into several parts after shrinking.
- New (off by default): `merge.core_ownership`, `training.include_negatives`, `training.select_by: f1`, `merge.simplify_tolerance`.
- New: the inference output keeps every raw detection with `tile_id, score, owned, touches_edge`, and it's resumable per row.
- New: GeoPackage end to end (no more 2 GB GeoJSON).
