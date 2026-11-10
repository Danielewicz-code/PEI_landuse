"""Training dataset + reproducible train/val split.

Expected layout (same as before):
    <dataset>/images/<name>.tif
    <dataset>/labels/<class_id>/<name>.tif   (instance-id raster, 0 = background)
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .preprocess import to_uint8_3ch


class ClearLandDataset(Dataset):
    def __init__(self, root, class_ids=(1,), image_ext=".tif", augment=True,
                 include_negatives=False, names: list[str] | None = None):
        """
        include_negatives: keep images that have NO label file (forest, town, water...).
            False = old behaviour. True lets the model learn what is NOT clear land.
        names: restrict to these file names (used for the frozen split).
        """
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.class_ids = tuple(class_ids)
        self.augment = augment

        all_names = sorted(f for f in os.listdir(self.images_dir) if f.lower().endswith(image_ext))
        if names is not None:
            wanted = set(names)
            all_names = [n for n in all_names if n in wanted]
        if not include_negatives:
            all_names = [n for n in all_names if self._has_label(n)]
        self.names = all_names

    def _has_label(self, name):
        return any((self.labels_dir / str(c) / name).exists() for c in self.class_ids)

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _read_mask(path):
        try:
            m = tifffile.imread(path)
            return m.max(axis=-1) if m.ndim == 3 else m
        except Exception:
            return None

    def __getitem__(self, idx):
        name = self.names[idx]
        img = to_uint8_3ch(tifffile.imread(self.images_dir / name))

        flip_h = self.augment and random.random() > 0.5
        flip_v = self.augment and random.random() > 0.5
        rot_k = random.randint(0, 3) if self.augment else 0

        def aug(a):
            if flip_h: a = np.fliplr(a)
            if flip_v: a = np.flipud(a)
            if rot_k: a = np.rot90(a, rot_k)
            return np.ascontiguousarray(a)

        img = aug(img)
        H, W = img.shape[:2]
        boxes, labels, inst = [], [], []

        for cid in self.class_ids:
            mp = self.labels_dir / str(cid) / name
            if not mp.exists():
                continue
            m = self._read_mask(mp)
            if m is None:
                continue
            m = aug(m)
            for uid in np.unique(m):
                if uid == 0:
                    continue
                single = (m == uid).astype(np.uint8)
                ys, xs = np.where(single)
                if xs.max() <= xs.min() or ys.max() <= ys.min():
                    continue
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                labels.append(int(cid))
                if single.shape != (H, W):
                    canvas = np.zeros((H, W), np.uint8)
                    canvas[: single.shape[0], : single.shape[1]] = single[:H, :W]
                    single = canvas
                inst.append(single)

        if boxes:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(inst), dtype=torch.uint8),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
            }
        return TF.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_split(dataset_root, split_file, val_fraction=0.15, seed=42, image_ext=".tif"):
    """Return (train_names, val_names). The split is written ONCE to `split_file`
    and re-used forever after, so val images never leak into training across
    sessions/fine-tunes, and both of you evaluate on the same images."""
    split_file = Path(split_file)
    if split_file.exists():
        s = json.loads(split_file.read_text())
        return s["train"], s["val"]
    names = sorted(f for f in os.listdir(Path(dataset_root) / "images") if f.lower().endswith(image_ext))
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = int(round(val_fraction * len(names)))
    split = {"seed": seed, "val_fraction": val_fraction, "val": sorted(names[:n_val]), "train": sorted(names[n_val:])}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2))
    return split["train"], split["val"]
