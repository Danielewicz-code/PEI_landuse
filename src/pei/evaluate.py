"""Real metrics (not just loss): per-field precision / recall / F1 and boundary IoU.

A predicted field matches a ground-truth field if their mask IoU >= iou_match.
Post-processing (mask threshold + erosion) is applied exactly like inference,
so the numbers reflect what ends up in the map.
"""
from __future__ import annotations

import numpy as np
import torch


def _postprocess_masks(pred, conf_t, mask_t, erosion_kernel, erosion_iter):
    import cv2
    keep = pred["scores"] >= conf_t
    masks = (pred["masks"][keep].squeeze(1) > mask_t).to(torch.uint8).cpu().numpy()
    if erosion_kernel:
        k = np.ones((erosion_kernel, erosion_kernel), np.uint8)
        masks = np.stack([cv2.erode(m, k, iterations=erosion_iter) for m in masks]) if len(masks) else masks
    return masks


def match_masks(pred_masks: np.ndarray, gt_masks: np.ndarray, iou_match=0.5):
    """Greedy one-to-one matching. Returns (tp, fp, fn, list_of_matched_ious)."""
    P, G = len(pred_masks), len(gt_masks)
    if P == 0 or G == 0:
        return 0, P, G, []
    p = pred_masks.reshape(P, -1).astype(np.float32)
    g = gt_masks.reshape(G, -1).astype(np.float32)
    inter = p @ g.T
    union = p.sum(1)[:, None] + g.sum(1)[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0)
    matched, used_p, used_g = [], set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < iou_match:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); matched.append(float(iou[i, j]))
    tp = len(matched)
    return tp, P - tp, G - tp, matched


@torch.no_grad()
def evaluate_model(model, dataset, device, cfg, max_images=None):
    icfg = cfg["inference"]
    model.eval()
    TP = FP = FN = 0
    ious = []
    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    for i in range(n):
        img, tgt = dataset[i]
        pred = model([img.to(device)])[0]
        pm = _postprocess_masks(pred, icfg["conf_threshold"], icfg["mask_threshold"],
                                icfg.get("erosion_kernel", 0), icfg.get("erosion_iterations", 1))
        tp, fp, fn, m = match_masks(pm, tgt["masks"].numpy(), cfg["evaluation"].get("iou_match", 0.5))
        TP += tp; FP += fp; FN += fn; ious += m
    prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
    return {
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "mean_matched_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "tp": TP, "fp": FP, "fn": FN, "images": n,
    }
