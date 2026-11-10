"""Model building + checkpoint I/O (one place, used by training, eval and inference)."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(num_classes: int = 2, pretrained_coco: bool = True,
                min_size: int = 800, max_size: int = 1333,
                detections_per_img: int = 100) -> torch.nn.Module:
    """Mask R-CNN ResNet50-FPN with a new head for `num_classes` (incl. background).

    NOTE on min_size/max_size: torchvision RESIZES every input so its short side
    is `min_size` (default 800). A 1024 tile is therefore shrunk to 800 before the
    network sees it. Defaults match the 1968 model; if you change them, retrain.
    """
    weights = "DEFAULT" if pretrained_coco else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights, min_size=min_size, max_size=max_size,
        box_detections_per_img=detections_per_img,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model


def build_model_from_cfg(cfg, pretrained_coco: bool = True):
    m = cfg["model"]
    return build_model(
        num_classes=m.get("num_classes", 2),
        pretrained_coco=pretrained_coco,
        min_size=m.get("min_size", 800),
        max_size=m.get("max_size", 1333),
        detections_per_img=m.get("detections_per_img", 100),
    )


def save_checkpoint(model, path, *, epoch, val_metric, optimizer=None, extra: dict | None = None):
    """Saves the .pth AND a human-readable .json next to it describing where it came from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_metric": val_metric}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    meta = {"file": path.name, "epoch": epoch, "val_metric": val_metric,
            "saved": _dt.datetime.now().isoformat(timespec="seconds"), **(extra or {})}
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    return ckpt if isinstance(ckpt, dict) else {}


def load_for_inference(cfg, device):
    model = build_model_from_cfg(cfg, pretrained_coco=False)
    load_checkpoint(model, cfg["model"]["checkpoint"], device)
    model.eval()
    return model
