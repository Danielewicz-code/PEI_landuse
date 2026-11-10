# pei-landuse

Clear-land (individual field) mapping for PEI, 1968 → 2020. Mask R-CNN, tiled inference and field separation, all in Colab.
**Read PROTOCOL.md before your first run.**

## First-time setup (once per person, all in Colab)
1. Get access to the private GitHub repo and the shared Drive folder `pei_landuse/`. The second person: add a shortcut to the shared folder in *My Drive* so the path is `/content/drive/MyDrive/pei_landuse`.
2. On GitHub, create a fine-grained token with read/write access to this repo. Put it in Colab ▸ Secrets (key icon) as `GITHUB_TOKEN` and allow notebook access.
3. Open `notebooks/00_git.ipynb` from GitHub in Colab (File ▸ Open notebook ▸ GitHub). Set your name at the top and run the clone cell. Your copy lands in `pei_landuse/code/<you>/pei-landuse`.
4. From now on, open notebooks from **your Drive copy**, and set `OPERATOR` in their first cell.
5. Move the old data into the new layout (once):

| Old (`clear_land_1968/…`) | New (`pei_landuse/…`) |
|---|---|
| `maps/whole_prince.tif` | `data/raw/1968/whole_prince.tif` |
| `training_data/first_training/` | `data/training/1968_first_training/` |
| `training_data/finetune_data/` | `data/training/1968_finetune/` |
| `model/*.pth` | `models/` |
| `geojsons/prince_raw_inference_backup.geojson` | `data/legacy/` |
| `geojsons/prince_clear_land_FINAL.gpkg` | `deliverables/clear_land_1968_prince.gpkg` |

## Everyday use
| Task | Notebook | Runtime | Config |
|---|---|---|---|
| Pull / branch / commit / push | `00_git` | CPU | none |
| Train / fine-tune | `01_train` | GPU | `train_*.yaml` |
| 2020 colour → model-ready gray | `02_prepare_raster` | CPU | `prepare_2020.yaml` |
| Map a county | `03_map` | GPU, then CPU | `infer_*.yaml` |
| Check a model (metrics + plots) | `04_review` | GPU | any training config |
| Fix merged fields on one spot | `05_tune_merge` | CPU | `remerge_*.yaml` / `infer_*.yaml` |

**Suggested first test:** run `03_map` with `remerge_1968_prince_hedgerow.yaml`. It re-separates the existing 1968 Prince detections with the new method; no GPU is needed. Then compare it in ArcGIS with the delivered 1968 layer at places where fields came out merged.

## Code map
```
src/pei/
  config.py      yaml loading (with `extends:`), run folders, logging
  model.py       build Mask R-CNN, save/load checkpoints (+ .json metadata)
  dataset.py     training dataset, frozen train/val split
  train.py       one loop for training and fine-tuning
  evaluate.py    precision / recall / F1 / IoU per field
  preprocess.py  tile normalisation, RGB→gray, resample, histogram match
  tiling.py      tile/block grid + ownership (tested)
  infer.py       resumable tiled inference -> runs/<id>/tiles/*.gpkg
  merge.py       dispatch + legacy 1968 merge (R-tree + watershed)
  segment.py     hedgerow-aware field separation (tested on synthetic fields)
  export.py      ids, area, county, GeoPackage + summary.json
  viz.py         GT vs prediction plots
tests/           python tests/test_tiling.py ; python tests/test_segment.py
```
