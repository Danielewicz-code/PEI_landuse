"""Side-by-side GT vs prediction plot (same as the 1968 notebook, config-driven)."""
from __future__ import annotations

import numpy as np


def plot_comparison(img_tensor, target, prediction, cfg, title=""):
    import cv2
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    icfg = cfg["inference"]
    img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    H, W = img.shape[:2]
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a in ax:
        a.imshow(img); a.axis("off")

    ov = np.zeros((H, W, 4), np.float32)
    for m in target["masks"].cpu().numpy():
        ov[m > 0] = [0, 0.8, 1, 0.35]
    ax[0].imshow(ov)
    ax[0].set_title(f"{title} ground truth ({len(target['masks'])})", fontweight="bold")

    scores = prediction["scores"].cpu().numpy()
    masks = prediction["masks"].squeeze(1).cpu().numpy()
    boxes = prediction["boxes"].cpu().numpy()
    k = icfg.get("erosion_kernel", 0)
    ov = np.zeros((H, W, 4), np.float32)
    n = 0
    for i in np.where(scores >= icfg["conf_threshold"])[0]:
        b = (masks[i] > icfg["mask_threshold"]).astype(np.uint8)
        if k:
            b = cv2.erode(b, np.ones((k, k), np.uint8), iterations=icfg.get("erosion_iterations", 1))
        if not b.any():
            continue
        n += 1
        ov[b > 0] = [0, 1, 0.2, 0.4]
        x0, y0, x1, y1 = boxes[i]
        ax[1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, lw=2, ec="red", fc="none"))
        ax[1].text(x0, max(y0 - 8, 15), f"{scores[i]*100:.1f}%", color="white", fontsize=9,
                   bbox=dict(facecolor="red", alpha=0.7, edgecolor="none", pad=2))
    ax[1].imshow(ov)
    ax[1].set_title(f"{title} predictions ({n})", fontweight="bold")
    plt.tight_layout(); plt.show()
