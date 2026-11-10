# Why distinct fields come out merged, and what changed

## In the 1968 merge
1. Mask R-CNN draws each mask on a coarse 28×28 grid stretched over the box, so neighbouring field masks often bleed across the hedgerow between them.
2. **Step 1 (R-tree merge) unions any two detections overlapping > 25 %** of the smaller one, measured as the max of the two ratios. Two different fields that bled into each other pass that test and become one polygon.
3. **Step 3 (watershed) uses those merged polygons as its seeds.** One merged polygon means one seed, so the watershed has nothing to split them with. Once step 1 merges two fields, nothing later can separate them.
4. Where the watershed does split, it runs on shape distance only, so the cut falls at the geometric middle of the blob rather than on the hedgerow.

## What `hedgerow_watershed` does differently (`src/pei/segment.py`)
- **Seeds are never unioned.** Every detection contributes its own shrunken core. Duplicates (tile overlaps, nested masks) are removed by priority instead: complete, high-score masks win, and a seed mostly covered by a better one is dropped.
- **The relief comes from the image.** It's built from dark thin lines (hedgerows, tree rows), light thin lines (roads, lanes) and edges, with a little shape distance as a tiebreak. Boundaries land on the hedgerow.
- **Re-joining is checked against the image.** Two touching segments are joined only if the line between them is weak, which fixes fields cut by tile edges. Two fields with a hedgerow between them stay apart.
- **The 2 m final shrink is kept** to strip the hedgerow pixels from each field, as before.
- **It runs in overlapping blocks**, each block keeping only the fields it owns. It uses all Colab CPU cores, resumes after a disconnect, and memory stays bounded however big the TIF is.

Tested on synthetic fields (`tests/test_segment.py`): with hedgerows from strong to faint and three noise levels, distinct fields always stay apart. Only in the hardest case (faint hedgerow plus heavy noise) does a tile-cut field stay split, which is the safe way to fail. Real imagery still needs tuning with `05_tune_merge`.

## Knobs, if the result is…
| You see | Try |
|---|---|
| Different fields still joined | lower `rag_threshold`, raise `w_dark` |
| One field cut into pieces | raise `rag_threshold` |
| Boundary off the hedgerow | `line_radius_px` ≈ half the hedgerow width in px; lower `w_dist` |
| Ghost slivers / duplicate seeds | raise `min_marker_keep` |
| Big fields cut at block seams | raise `block_halo` (> half the largest field) |

## Other ideas, later
- Train with `model.min_size: 1024` so tiles aren't shrunk to 800 px internally. Needs retraining; gives sharper masks.
- Select checkpoints by F1 (`training.select_by: f1`) instead of validation loss.
- For 2020: label about 20–30 prepared tiles, run `04_review`, and fine-tune if F1 drops a lot compared with 1968.
