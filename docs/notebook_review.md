# Review of the 1968 notebook + improvement roadmap

## Why the borders are blobby (root cause)
Mask R-CNN predicts each mask as a **28×28 grid** that is stretched to fit the box. For a field 500 px wide, one mask cell covers about 18 px. The edges can't be sharper than that, so masks bleed across hedgerows into neighbouring fields. On top of that, torchvision shrinks each 1024 tile to 800 px (`min_size`) before the network sees it.

Everything after the model is compensating for this:
- mask threshold 0.65, then 5×5 erosion
- union of overlapping detections, which can join two neighbouring fields that bled into each other
- watershed to split them again; it uses only the shapes, not the image, so the split lands at the geometric middle rather than on the real hedgerow
- a final −2 m shrink

**Side effect: every field edge is pulled in by about 3–4 m in total, so total clear-land area is biased low.** For a 1968 vs 2020 comparison the bias roughly cancels if both years use the same pipeline. It doesn't cancel for absolute hectares.

## Roadmap (each item goes behind a config flag and gets compared on the frozen split)
1. **`core_ownership`** (built, off by default): removes overlap duplicates before merging. It should also cut merge time a lot.
2. **Higher-resolution masks**: train with `min_size: 1024` so tiles aren't shrunk. Requires retraining.
3. **Boundary-aware semantic model** (biggest change): a U-Net that predicts *field interior / boundary / other* per pixel. In overlaps you average the probabilities, which makes seams disappear. Fields are then the connected interiors, with watershed run on the *predicted boundary* rather than on shape distance. This is the standard approach in field-delineation work and suits hedgerow landscapes like PEI.
4. **Hard negatives**: `include_negatives: true` plus a few tiles of forest, towns, golf courses and water. 2020 has more of these than 1968.
5. **Select checkpoints by F1** (`select_by: f1`), not loss.
6. **2020 domain check**: label about 20–30 tiles of prepared 2020 and run `04_review`. If F1 drops a lot compared with 1968, fine-tune on them.
7. **Change prior**: flag 2020 fields with low overlap with 1968 fields for manual review.
