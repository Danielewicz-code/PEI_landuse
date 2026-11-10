# Working protocol

These are the rules for the two of us. They're short on purpose. Everything runs in Colab, including git. If a rule stops making sense, change it here and note it in CHANGELOG.md.

## 1. Where things live

| What | Where | In git? |
|---|---|---|
| Code, configs, notebooks (no outputs) | private GitHub repo `pei-landuse` | yes |
| Your working copy of the code | Drive `pei_landuse/code/<you>/pei-landuse` | it *is* the clone |
| Rasters, labels, models, results | Drive `pei_landuse/` | **never** |
| Decisions and changes | `CHANGELOG.md` | yes |

Each of us has our own clone under `code/<name>/`, so we never edit the same files at the same time. Changes move between us through GitHub (`00_git.ipynb`), not by editing each other's folder. The clone lives on Drive, so edits made in Colab's file editor survive disconnects.

Drive layout (the `paths:` block in `configs/base.yaml` points here):

```
pei_landuse/
├── code/<name>/pei-landuse/         each person's git clone
├── data/
│   ├── raw/1968/  raw/2020/         originals, READ-ONLY, never overwritten
│   ├── interim/2020/                prepared rasters (grayscale, resampled)
│   ├── training/<dataset>/          images/, labels/1/, split.json
│   ├── legacy/                      old outputs we still compare against
│   └── reference/                   county boundaries etc.
├── models/                          <name>.pth + <name>.json (where it came from)
├── runs/<YYYY-MM-DD>_<name>/        one folder per run, see §3
└── deliverables/                    final layers we hand over (copied from runs/)
```

## 2. Sensitive data

- The repo is **private**. Only the two of us have access, and `.gitignore` blocks rasters, vectors and models.
- **Notebook outputs contain map images.** `00_git.ipynb` installs nbstripout in your clone so outputs are stripped on commit. Don't remove it.
- The Drive folder is shared only with the two of us. No "anyone with the link".
- The GitHub token goes in Colab ▸ Secrets (`GITHUB_TOKEN`). It's never written in a cell or saved in `.git/config`.
- Don't paste map screenshots or file contents into public tools or forums.

## 3. Runs

- **Every run starts from a config file.** No editing numbers inside notebooks. To change a setting, copy the config, change it, and give it a new `run.name`. The one exception is the `TRY = {}` box in `05_tune_merge`. It's for tuning, and the values that work get copied into a config.
- `make_run_dir` creates `runs/<date>_<name>/` and freezes `config.yaml` plus `run_meta.json` (git commit, operator) into it. Results go only into that folder.
- Same name on the same day means the same folder. That's how resume works after a Colab disconnect: re-run and finished rows or blocks are skipped. **For a new experiment, use a new name.**
- If the commit hash in `run_meta.json` ends in `-dirty`, the run used uncommitted code. That's fine for trying things. **Deliverables must come from a clean commit.**
- Inference (GPU) and field separation (CPU) are separate stages. To try new separation settings, run only Stage 2 of `03_map` or use `05_tune_merge`. The GPU work is never redone.

## 4. Changing code

- `main` always works. For a change: `00_git` → new branch → edit → commit + push → open a PR on GitHub → the other person skims it → merge. Small fixes can be merged by the author.
- Pull before starting work (`00_git`, status/pull cell).
- **Changing default behaviour goes through a config option.** Add the new behaviour as an option with the old behaviour as the default, test it, and only then switch the default. Write one line in CHANGELOG.md.
- Logic goes in `src/pei/`. Notebooks only load a config and call functions. If a notebook cell is longer than about 15 lines, that code belongs in `src/`. Notebooks use autoreload, so edits to `src/` take effect without restarting.
- After touching tiling or segmentation, run the tests: `!python tests/test_tiling.py` and `!python tests/test_segment.py`.

## 5. Models

- File names: `pei_<year>_<what>_v<N>.pth`, e.g. `pei_2020_finetune_v1.pth`. Never overwrite a model that has been used for a deliverable; bump the version instead.
- Every `.pth` gets a `.json` beside it (written automatically) recording the dataset, starting weights, run and score.
- **Compare models only on the frozen split** (`split.json` in the dataset folder, written once). Don't delete or regenerate it. If the dataset gets new images, create a new dataset folder.

## 6. Naming

- Datasets: `<year>_<purpose>`, e.g. `1968_finetune`, `2020_labels_v1`
- Runs: `<year>_<county>_<variant>`, e.g. `2020_prince_v1`, `1968_prince_hedgerow`
- Final layers: `clear_land_<year>_<runname>.gpkg`, layer `clear_land`. Fields: `field_id, year, area_m2, county, run_id`
- Formats: rasters as tiled GeoTIFF, vectors as **GeoPackage** (no GeoJSON or shapefile for anything big)
- CRS: one CRS for everything, set in `project.crs`

## 7. Handover checklist (before a layer leaves `runs/`)

- [ ] Made from a clean commit (no `-dirty`)
- [ ] Spot-checked in ArcGIS on at least 3 areas, including a block seam and a spot that used to come out merged
- [ ] `summary.json` areas look sane against the previous year or county
- [ ] One line in CHANGELOG.md: what it is, which run, known issues
- [ ] Copied to `deliverables/`
