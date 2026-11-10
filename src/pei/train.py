"""ONE training loop for both 'train from COCO' and 'fine-tune from a checkpoint'.

Which one happens is decided by config `training.init_from`:
    coco                       -> fresh head on COCO weights
    /path/to/checkpoint.pth    -> continue from that model (fine-tune)
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import log
from .dataset import ClearLandDataset, collate_fn, get_split
from .evaluate import evaluate_model
from .model import build_model_from_cfg, load_checkpoint, save_checkpoint


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _val_loss(model, loader, device):
    model.train()  # torchvision detection models only return losses in train mode (BN is frozen)
    total, comps = 0.0, {}
    for imgs, tgts in loader:
        imgs = [i.to(device) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            ld = model(imgs, tgts)
        total += sum(ld.values()).item()
        for k, v in ld.items():
            comps[k] = comps.get(k, 0.0) + v.item()
    n = max(len(loader), 1)
    return total / n, {k: v / n for k, v in comps.items()}


def train(cfg, run_dir, device):
    t = cfg["training"]
    seed_everything(t.get("seed", 42))
    ds_root = Path(cfg["paths"]["training_data"]) / t["dataset"]
    split_file = ds_root / "split.json"
    train_names, val_names = get_split(ds_root, split_file, t.get("val_fraction", 0.15), t.get("seed", 42))
    log(run_dir, f"Split file: {split_file}")

    cls = tuple(t.get("class_ids", [1]))
    neg = t.get("include_negatives", False)
    train_ds = ClearLandDataset(ds_root, cls, augment=True, include_negatives=neg, names=train_names)
    val_ds = ClearLandDataset(ds_root, cls, augment=False, include_negatives=False, names=val_names)
    log(run_dir, f"Train {len(train_ds)} | Val {len(val_ds)}")

    dl_kw = dict(batch_size=t.get("batch_size", 2), num_workers=t.get("num_workers", 2), collate_fn=collate_fn)
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    init = t.get("init_from", "coco")
    model = build_model_from_cfg(cfg, pretrained_coco=(init == "coco"))
    if init != "coco":
        load_checkpoint(model, init, device)
        log(run_dir, f"Initialised from {init}")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=t["lr_step"], gamma=t["lr_gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    select_by = t.get("select_by", "val_loss")  # "val_loss" (old behaviour) or "f1"
    best = float("inf") if select_by == "val_loss" else -1.0
    out_path = Path(cfg["paths"]["models"]) / f"{t['output_name']}.pth"
    history = []

    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        model.train()
        tr = 0.0
        for imgs, tgts in train_dl:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t_.items()} for t_ in tgts]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = sum(model(imgs, tgts).values())
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr += loss.item()
        sched.step()
        tr /= max(len(train_dl), 1)
        vl, comps = _val_loss(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": tr, "val_loss": vl, **comps}

        if select_by == "f1" or epoch == t["epochs"]:
            rec.update(evaluate_model(model, val_ds, device, cfg))

        score = vl if select_by == "val_loss" else rec.get("f1", -1)
        improved = score < best if select_by == "val_loss" else score > best
        history.append(rec)
        log(run_dir, f"ep {epoch:02d} {time.time() - t0:.0f}s train {tr:.4f} val {vl:.4f} "
                     f"mask {comps.get('loss_mask', 0):.4f}" + (f" f1 {rec['f1']:.3f}" if "f1" in rec else "")
                     + ("  ★ saved" if improved else ""))
        if improved:
            best = score
            save_checkpoint(model, out_path, epoch=epoch, val_metric={select_by: score}, optimizer=opt,
                            extra={"run_id": cfg["run"]["id"], "dataset": t["dataset"], "init_from": init,
                                   "split_file": str(split_file)})
        (Path(run_dir) / "train_history.json").write_text(json.dumps(history, indent=2))

    log(run_dir, f"Best model: {out_path} ({select_by}={best:.4f})")
    return out_path
