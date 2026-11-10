# Working protocol

These are the rules for the two of us. They're short on purpose. If one stops making sense, change it here and note it in CHANGELOG.md.

## 1. Where things live

| What | Where | In git? |
|---|---|---|
| Code, configs, notebooks (no outputs) | private GitHub repo `pei-landuse` | yes |
| Rasters, labels, models, results | shared Drive folder `pei_landuse/` | **never** |
| Decisions and changes | `CHANGELOG.md` | yes |

Drive layout (the `paths:` block in `configs/base.yaml` points here):

```
pei_landuse/
├── data/
│   ├── raw/1968/  raw/2020/         originals, READ-ONLY, never overwritten
│   ├── interim/2020/                prepared rasters (grayscale, resampled)
│   ├── training/<dataset>/          images/, labels/1/, split.json
│   └── reference/                   county boundaries etc.
├── models/                          <name>.pth + <name>.json (where it came from)
├── runs/<YYYY-MM-DD>_<name>/        one folder per run, see §3
└── deliverables/                    the final layers we hand over (copied from runs/)
```

## 2. Sensitive data

- The repo is **private**. Only the two of us have access. Data never goes in git; `.gitignore` blocks rasters, vectors and models.
- **Notebook outputs contain map images, so strip them before committing.** Do this once on each machine:
  `pip install nbstripout && nbstripout --install` (inside the repo).
- The Drive folder is shared only with the two of us. No "anyone with the link".
- Tokens (GitHub) go in Colab ▸ Secrets, never in a cell.
- Don't paste map screenshots or file contents into public tools or forums.

## 3. Runs

- **Every run starts from a config file.** No editing numbers inside notebooks. To change a threshold, copy the config, change it, and give it a new `run.name`.
- `make_run_dir` creates `runs/<date>_<name>/` and freezes `config.yaml` plus `run_meta.json` (git commit, operator) into it. Results go only into that folder.
- Same name on the same day means the same folder. That's how resume works: re-run and finished stages are skipped. **For a new experiment, use a new name.**
- If the commit hash in `run_meta.json` ends in `-dirty`, the run used uncommitted code. That's fine for trying things. **Deliverables must come from a clean commit.**
- Merging is separate from inference. To try different merge settings, re-run only Stage 2 in `03_map.ipynb`.

## 4. Changing code

- `main` always works. Do new work on a branch (`feat/core-ownership`, `fix/nodata`), then open a PR. The other person skims it, or you merge it yourself if it's small.
- **Changing default behaviour goes through a config flag.** Add the option with the old behaviour as the default, test it, and only then flip the default. Write one line in CHANGELOG.md.
- Logic goes in `src/pei/`. Notebooks only load a config and call functions. If a notebook cell is longer than about 15 lines, that code belongs in `src/`.
- Run `python tests/test_tiling.py` after touching tiling.

## 5. Models

- File names: `pei_<year>_<what>_v<N>.pth`, e.g. `pei_2020_finetune_v1.pth`. Never overwrite a model that has been used for a deliverable; bump the version instead.
- Every `.pth` gets a `.json` beside it (written automatically) recording the dataset, starting weights, run and score.
- **Compare models only on the frozen split** (`split.json` in the dataset folder, written once). Don't delete or regenerate it. If the dataset gets new images, create a new dataset folder.

## 6. Naming

- Datasets: `<year>_<purpose>`, e.g. `1968_finetune`, `2020_labels_v1`
- Runs: `<year>_<county>_<variant>`, e.g. `2020_prince_v1`, `2020_prince_coreown`
- Final layers: `clear_land_<year>_<runname>.gpkg`, layer `clear_land`. Fields: `field_id, year, area_m2, county, run_id`
- Formats: rasters as tiled GeoTIFF, vectors as **GeoPackage only** (no GeoJSON or shapefile for anything big)
- CRS: one CRS for everything, set in `project.crs`

## 7. Handover checklist (before a layer leaves `runs/`)

- [ ] Made from a clean commit (no `-dirty`)
- [ ] Spot-checked in ArcGIS on at least 3 areas, including one tile seam
- [ ] `summary.json` areas look sane against the previous year or county
- [ ] One line in CHANGELOG.md: what it is, which run, known issues
- [ ] Copied to `deliverables/`
