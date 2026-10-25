"""Pure numpy/OpenCV mask operations. Masks are uint8 HxW (0..255), images uint8 HxWx3 RGB."""
import cv2
import numpy as np

WORK = 2048  # long side at which guided-filter coefficients are solved


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def guided_upsample(mask_lr, rgb, model_res, eps=1e-3):
    """Upsample a low-res soft mask (float 0..1) to rgb's size using the image as guide.

    Fast guided filter: coefficients are solved at <= WORK px, then applied at full res,
    so edges follow the full-resolution image.
    ponytail: grayscale guide; a colour guide helps where fg/bg share luminance.
    """
    H, W = rgb.shape[:2]
    s = min(1.0, WORK / max(H, W))
    w, h = max(1, round(W * s)), max(1, round(H * s))
    small = rgb if s == 1.0 else cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    I = gray(small).astype(np.float32) / 255
    p = cv2.resize(mask_lr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    r = max(2, round(2 * max(w, h) / model_res))
    mI, mp = _box(I, r), _box(p, r)
    cov = _box(I * p, r) - mI * mp
    var = _box(I * I, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    a, b = _box(a, r), _box(b, r)
    if s < 1.0:
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)
        I = gray(rgb).astype(np.float32)
        I *= 1 / 255
    a *= I
    a += b
    return to_u8(a)


def shift_confidence(p, bias):
    """Move the 0.5 cut of a 0..1 probability map by bias logits: > 0 takes in more, < 0 less.

    Works in logit space because BiRefNet output is nearly binary: few pixels are between 0.01 and 0.99.
    Far from the edge the logits change slowly, so the shifted map is steepened by 1 + |bias| to keep edges sharp.
    """
    if not bias:
        return p
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.clip((np.log(p / (1 - p)) + bias) * (1 + abs(bias)), -30, 30)
    return 1 / (1 + np.exp(-z))


def to_u8(x):
    """float 0..1 -> uint8, in place where possible."""
    x *= 255
    np.clip(x, 0, 255, out=x)
    return x.astype(np.uint8)


def combine(base, new, op):
    if base is None or op == "Replace":
        return new
    if op == "Add":
        return np.maximum(base, new)
    if op == "Intersect":
        return np.minimum(base, new)
    if op == "Subtract":
        return cv2.multiply(base, cv2.bitwise_not(new), scale=1 / 255)
    raise ValueError(op)


def bbox(mask, thresh=8):
    """(x0, y0, x1, y1) of pixels above thresh, or None."""
    ys = np.flatnonzero((mask > thresh).any(axis=1))
    if not len(ys):
        return None
    xs = np.flatnonzero((mask[ys[0]:ys[-1] + 1] > thresh).any(axis=0))
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def pad_box(box, pad, W, H):
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


# ---------- colour key ----------

def color_distance(rgb, x, y):
    """CIE76 delta-E (Lab) of every pixel to the 5x5 mean colour around (x, y)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)  # uint8 Lab: L scaled to 0..255, a/b offset by 128
    H, W = lab.shape[:2]
    patch = lab[max(0, y - 2):min(H, y + 3), max(0, x - 2):min(W, x + 3)].reshape(-1, 3)
    ref = patch.astype(np.float32).mean(0)
    scale = (100 / 255, 1.0, 1.0)
    d = np.zeros((H, W), np.float32)
    for c in range(3):
        t = lab[..., c].astype(np.float32)
        t -= ref[c]
        t *= scale[c]
        t *= t
        d += t
    np.sqrt(d, out=d)
    return d


def key_alpha(dist, tol, soft, invert=False):
    """Pixels within tol of the key colour become transparent, with a smoothstep ramp of width soft."""
    t = dist - tol
    t *= 1 / max(soft, 1e-3)
    np.clip(t, 0, 1, out=t)
    t = t * t * (3 - 2 * t)
    a = to_u8(t)
    return cv2.bitwise_not(a) if invert else a


# ---------- refinement / finishing ----------

def band(mask, reach):
    """(outer, core): mask dilated by reach px / eroded by reach/3 px. Morphology runs at <=1024 px (big kernels are slow).

    Refinement may add detail up to reach px outside the mask but only remove within reach/3 inside it,
    so a matte that disagrees with the selection cannot eat into it.
    """
    H, W = mask.shape
    f = max(1.0, max(H, W) / 1024)
    small = mask if f == 1 else cv2.resize(mask, (round(W / f), round(H / f)), interpolation=cv2.INTER_AREA)
    r = max(1, round(reach / f))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    outer = cv2.GaussianBlur(cv2.dilate(small, k), (0, 0), max(0.5, r / 4))
    ri = max(1, r // 3)
    core = cv2.erode(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ri + 1, 2 * ri + 1)))
    if f > 1:
        outer = cv2.resize(outer, (W, H), interpolation=cv2.INTER_LINEAR)
        core = cv2.resize(core, (W, H), interpolation=cv2.INTER_LINEAR)
    return outer, core


def grow(mask, px):
    if px == 0:
        return mask
    r = abs(px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask, k) if px > 0 else cv2.erode(mask, k)


def feather(mask, px):
    return mask if px <= 0 else cv2.GaussianBlur(mask, (0, 0), px)


def _sample(img, ys, xs, f):
    """Bilinear sample a downscaled map at full-res pixel coords."""
    n, cols = len(ys), 4096  # remap maps must stay under 32767 rows/cols
    pad = -n % cols
    mx = np.pad((xs + 0.5) / f - 0.5, (0, pad)).astype(np.float32).reshape(-1, cols)
    my = np.pad((ys + 0.5) / f - 0.5, (0, pad)).astype(np.float32).reshape(-1, cols)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, img.shape[2])[:n]


def _spread(rgb_masked, w, f, sigma):
    """Normalised-convolution colour fill at 1/f scale: average colour of the weighted pixels nearby."""
    H, W = w.shape
    size = (max(1, W // f), max(1, H // f))
    num = cv2.resize(rgb_masked, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    den = cv2.resize(w.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    out = num.copy()
    have = den.copy()
    for s in (sigma, sigma * 4, sigma * 16):  # widen until every pixel has a colour
        n = cv2.GaussianBlur(num, (0, 0), s)
        d = cv2.GaussianBlur(den, (0, 0), s)
        fill = (have < 1e-3) & (d > 1e-4)
        out[fill] = n[fill] / d[fill, None]
        have[fill] = 1
    rest = have < 1e-3  # nothing weighted within reach: global mean colour, not black
    out[rest] = num.sum((0, 1)) / max(den.sum(), 1e-3)
    ok = den > 1e-3
    out[ok] = num[ok] / den[ok, None]
    return out


def defringe(rgb, alpha, strength):
    """Remove background colour bleed from semi-transparent edge pixels.

    Estimates local foreground F and background B colours, then solves I = aF + (1-a)B for F.
    Only edge pixels are touched, so memory stays proportional to the edge, not the image.
    """
    if strength <= 0:
        return rgb
    ys, xs = np.nonzero((alpha > 2) & (alpha < 250))
    if not len(ys):
        return rgb
    H, W = alpha.shape
    f = max(1, round(max(H, W) / 1024))
    sigma = max(1.0, max(H, W) / 200 / f)  # ~0.5% of the image, in low-res px
    fg = (alpha >= 250).astype(np.uint8)
    bg = (alpha <= 2).astype(np.uint8)
    Fm = _spread(rgb * fg[..., None], fg, f, sigma)
    Bm = _spread(rgb * bg[..., None], bg, f, sigma)
    F = _sample(Fm, ys, xs, f)
    B = _sample(Bm, ys, xs, f)
    I = rgb[ys, xs].astype(np.float32)
    a = alpha[ys, xs].astype(np.float32)[:, None] / 255
    solved = np.clip((I - (1 - a) * B) / np.maximum(a, 1e-3), 0, 255)
    est = np.where(a >= 0.25, solved, F)
    out = rgb.copy()
    out[ys, xs] = np.clip(I + (est - I) * strength, 0, 255).astype(np.uint8)
    return out


def finish(rgb, mask, grow_px=0, feather_px=0.0, defringe_amt=0.0, scale=1.0):
    """Apply shrink/grow, feather and defringe. Pixel params are in full-res pixels; scale = this res / full."""
    a = grow(mask, round(grow_px * scale))
    a = feather(a, feather_px * scale)
    return defringe(rgb, a, defringe_amt), a


if __name__ == "__main__":
    # self-check on a synthetic image: red disc on blue, soft edge
    H, W = 300, 400
    yy, xx = np.mgrid[:H, :W]
    d = np.hypot(xx - 200, yy - 150)
    a_true = np.clip((80 - d) / 6 + 0.5, 0, 1)
    rgb = (a_true[..., None] * [220, 30, 30] + (1 - a_true[..., None]) * [20, 40, 230]).astype(np.uint8)

    lr = cv2.resize(a_true.astype(np.float32), (64, 48), interpolation=cv2.INTER_AREA)
    up = guided_upsample(lr, rgb, 64)
    assert up.shape == (H, W) and abs(up.astype(float) / 255 - a_true).mean() < 0.02, "guided upsample"

    p = np.array([1e-4, 0.3, 0.5, 0.9, 1.0], np.float32)
    assert (shift_confidence(p, 0) == p).all()
    assert ((shift_confidence(p, 3) > 0.5) == [0, 1, 1, 1, 1]).all(), "positive bias takes in more"
    assert ((shift_confidence(p, -3) > 0.5) == [0, 0, 0, 0, 1]).all(), "negative bias keeps less"
    assert shift_confidence(p, 10)[0] > 0.5 > shift_confidence(p, -10)[3]

    dist = color_distance(rgb, 10, 10)
    k = key_alpha(dist, 10, 10)
    assert k[10, 10] == 0 and k[150, 200] == 255, "colour key"
    assert key_alpha(dist, 10, 10, invert=True)[10, 10] == 255

    m = (a_true * 255).astype(np.uint8)
    full = np.full_like(m, 255)
    assert (combine(None, m, "Add") == m).all()
    assert combine(full, m, "Subtract")[150, 200] == 0 and combine(full, m, "Subtract")[0, 0] == 255
    assert (combine(m, full, "Intersect") == m).all()
    assert bbox(m) is not None and bbox(np.zeros_like(m)) is None

    out = defringe(rgb, m, 1.0)
    edge = (m > 60) & (m < 200)
    assert out[edge][:, 2].mean() < rgb[edge][:, 2].mean() - 40, "defringe removes blue bleed"
    assert grow(m, 5).sum() > m.sum() > grow(m, -5).sum()

    # f = 2 (long side >= 1536): low-res weights must keep fractional coverage, or edge colour halves
    yy, xx = np.mgrid[:1600, :2000]
    a2 = np.clip((530 - np.hypot(xx - 1000, yy - 800)) / 3 + 0.5, 0, 1)
    rgb2 = (a2[..., None] * [220, 30, 30] + (1 - a2[..., None]) * [20, 40, 230]).astype(np.uint8)
    m2 = (a2 * 255).astype(np.uint8)
    out2, edge2 = defringe(rgb2, m2, 1.0), (m2 > 60) & (m2 < 200)
    assert abs(out2[edge2][:, 0].mean() - 220) < 10, "defringe f=2 keeps red"
    assert abs(out2[edge2][:, 2].mean() - 30) < 10 < rgb2[edge2][:, 2].mean() - 30, "defringe f=2 removes blue"
    print("maskops ok")
