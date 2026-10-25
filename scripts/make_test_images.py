"""Generate synthetic test images into test_images/: soft-edged subject with hair strands, and a ~50 MP stress image."""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "test_images"


def scene(w, h, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky = (1 - t) * np.array([120, 170, 235], np.float32) + t * np.array([215, 230, 250], np.float32)
    img = np.broadcast_to(sky, (h, w, 3)).copy()
    cx, cy, r = w // 2, int(h * 0.55), int(min(w, h) * 0.25)
    cv2.circle(img, (cx, cy), r, (200, 90, 40), -1, cv2.LINE_AA)  # orange "head"
    cv2.rectangle(img, (cx - r // 2, cy + r - 5), (cx + r // 2, h), (60, 60, 140), -1, cv2.LINE_AA)
    for _ in range(400):  # thin wavy strands for matting
        a = rng.uniform(np.pi * 1.05, np.pi * 1.95)
        x0, y0 = cx + r * 0.9 * np.cos(a), cy + r * 0.9 * np.sin(a)
        length = rng.uniform(0.15, 0.45) * r
        pts = [(x0 + k * length / 20 * np.cos(a) + np.sin(k / 3) * 3, y0 + k * length / 20 * np.sin(a)) for k in range(21)]
        cv2.polylines(img, [np.int32(pts)], False, (120, 50, 20), max(1, w // 3000), cv2.LINE_AA)
    return img.astype(np.uint8)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, (w, h) in {"strands_small.png": (1600, 1200), "strands_50mp.jpg": (8660, 5774)}.items():
        cv2.imwrite(str(OUT / name), cv2.cvtColor(scene(w, h), cv2.COLOR_RGB2BGR))
        print("wrote", OUT / name)
