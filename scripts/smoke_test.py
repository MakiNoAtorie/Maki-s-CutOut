"""Headless check of every engine path on one image. Usage: uv run python scripts/smoke_test.py IMAGE [OUT_DIR]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine.device  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from engine import maskops  # noqa: E402
from engine.device import device_label  # noqa: E402
from engine.models import Engine  # noqa: E402


def save(out, name, rgb, a):
    cv2.imwrite(str(out / name), cv2.cvtColor(np.dstack([rgb, a]), cv2.COLOR_RGBA2BGRA))


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "smoke_out")
    out.mkdir(exist_ok=True)
    rgb = np.asarray(Image.open(src).convert("RGB")).copy()
    H, W = rgb.shape[:2]
    print(f"{src.name} {W}x{H} on {device_label()}")
    eng = Engine()

    t = time.time()
    m = eng.subject(rgb)
    print(f"subject: {time.time() - t:.1f}s  coverage {(m > 127).mean():.2f}")
    assert m.shape == (H, W)
    save(out, "subject.png", rgb, m)
    t = time.time()
    eng.subject(rgb)
    print(f"subject warm: {time.time() - t:.1f}s")

    box = maskops.bbox(m) or (0, 0, W, H)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    t = time.time()
    s1 = eng.sam(1, rgb, [(cx, cy)], [1])
    t1 = time.time() - t
    t = time.time()
    s2 = eng.sam(1, rgb, [(cx, cy)], [1], box)
    print(f"sam: first {t1:.1f}s (load+embed), cached {time.time() - t:.2f}s")
    assert s1.shape == (H, W) and (s1 > 127).any() and (s2 > 127).any()
    save(out, "sam_point.png", rgb, s1)

    t = time.time()
    r = eng.refine(rgb, m)
    print(f"refine: {time.time() - t:.1f}s")
    t = time.time()
    eng.refine(rgb, m)
    print(f"refine warm: {time.time() - t:.1f}s")
    frgb, fa = maskops.finish(rgb, r, defringe_amt=1.0)
    save(out, "refined_defringed.png", frgb, fa)

    t = time.time()
    k = maskops.key_alpha(maskops.color_distance(rgb, W // 2, 10), 12, 10)
    print(f"color key: {time.time() - t:.2f}s")
    save(out, "colorkey.png", rgb, k)
    t = time.time()
    rect = (W // 2, 0, W, int(H * 0.7))  # right half / top: the tree in the landscape photos
    mr = eng.subject(rgb, rect)
    print(f"subject in rect: {time.time() - t:.1f}s  coverage {(mr > 127).mean():.2f}")
    save(out, "subject_rect.png", *maskops.finish(rgb, eng.refine(rgb, mr), defringe_amt=1.0))
    region = np.zeros((H, W), np.uint8)  # painted blob over the left part (left tree in the landscape photos)
    cv2.ellipse(region, (W // 5, H // 2), (W // 4, H // 4), 0, 0, 360, 255, -1)
    f = min(1.0, 2048 / max(H, W))  # the app paints at proxy res (viewer.PROXY)
    paint = cv2.resize(region, (round(W * f), round(H * f)), interpolation=cv2.INTER_AREA)
    t = time.time()
    mp = eng.paint(rgb, paint)
    out_d = cv2.distanceTransform(255 - region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    in_d = cv2.distanceTransform(region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inner = (mp[in_d > 3 / f] > 127).mean()
    print(f"paint region: {time.time() - t:.1f}s  coverage {(mp > 127).mean():.2f}  inner {inner:.3f}")
    assert not mp[out_d > max(4, round(0.02 * max(H, W))) + 3 / f].any(), "result must stay within reach of the paint"
    assert inner >= 0.9, "the painted area is the selection"
    save(out, "paint.png", rgb, mp)
    if (k > 127).any():
        t = time.time()
        kr = eng.refine(rgb, k)
        print(f"refine color key: {time.time() - t:.1f}s")
        save(out, "colorkey_refined.png", *maskops.finish(rgb, kr, defringe_amt=1.0))
    print("smoke ok ->", out)


if __name__ == "__main__":
    main()
