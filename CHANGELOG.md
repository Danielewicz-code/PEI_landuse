# Changelog

One line per change that affects results: what changed, why, and which runs are affected.
Newest first.

## 2026-11-10 — v0.1.0: notebook → package
- Moved `clearland.ipynb` into `src/pei/` + configs. `merge.method: legacy` reproduces the 1968 pipeline exactly: same thresholds, black-pixel tile skip, R-tree merge, watershed, 2 m hedgerow shrink.
- Config records `mask_threshold: 0.65`, the value the 1968 inference loop actually used.
- Train/val split is now frozen in `split.json`, written once per dataset, so fine-tuning and evaluation always use the same validation images.
- New: `merge.method: hedgerow_watershed` (`pei/segment.py`) to stop distinct fields from being merged. Seeds are never unioned, boundaries follow hedgerows and roads in the image, and tile-cut fields are re-joined only where no line separates them. The 2020 config uses it; 1968 stays on legacy until compared (`configs/remerge_1968_prince_hedgerow.yaml`).
- New: inference output keeps every raw detection with `tile_id, score, owned, touches_edge`. Inference and field separation can resume after a disconnect.
- New: GeoPackage end to end; git from Colab (`00_git`); tuning notebook (`05_tune_merge`).
