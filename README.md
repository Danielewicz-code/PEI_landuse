# pei-landuse

Clear-land (field) mapping for PEI, 1968 → 2020. Mask R-CNN + tiled inference + vector merge.
**Read PROTOCOL.md before your first run.**

## First-time setup (once per person)
1. Get access to the private repo and the shared Drive folder `pei_landuse/`.
2. Create a GitHub fine-grained token with read access to this repo. Put it in Colab ▸ Secrets as `GITHUB_TOKEN`.
3. On your laptop, in the repo: `pip install nbstripout && nbstripout --install`.
4. Move the old data into the new layout (once):

| Old (`clear_land_1968/…`) | New (`pei_landuse/…`) |
|---|---|
| `maps/whole_prince.tif` | `data/raw/1968/whole_prince.tif` |
| `training_data/first_training/` | `data/training/1968_first_training/` |
| `training_data/finetune_data/` | `data/training/1968_finetune/` |
| `model/*.pth` | `models/` |
| `geojsons/prince_clear_land_FINAL.gpkg` | `deliverables/clear_land_1968_prince.gpkg` |

## Everyday use
| Task | Notebook | Config |
|---|---|---|
| Train / fine-tune | `01_train` | `configs/train_*.yaml` |
| 2020 colour → model-ready gray | `02_prepare_raster` | `configs/prepare_2020.yaml` |
| Map a county | `03_map` | `configs/infer_*.yaml` |
| Check a model (metrics + plots) | `04_review` | any training config |

To start a new run, copy a config, change `run.name` and whatever else differs, then put its path in the notebook's first cell.

## Code map
```
src/pei/
  config.py      yaml loading (with `extends:`), run folders, logging
  model.py       build Mask R-CNN, save/load checkpoints (+ .json metadata)
  dataset.py     training dataset, frozen train/val split
  train.py       one loop for training and fine-tuning
  evaluate.py    precision / recall / F1 / IoU per field
  preprocess.py  tile normalisation, RGB→gray, resample, histogram match
  tiling.py      tile grid + core ownership (tested)
  infer.py       resumable tiled inference -> runs/<id>/tiles/*.gpkg
  merge.py       R-tree merge + watershed (1968 logic) -> clean fields
  export.py      ids, area, county, GeoPackage + summary.json
  viz.py         GT vs prediction plots
```
