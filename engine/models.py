"""Model loading and inference. Only one model lives on the device at a time.

All methods are meant to be called from a single worker thread.
Both model families load through transformers: BiRefNet via trust_remote_code (pinned revision),
SAM 2.1 natively (Sam2Model), so no extra sam2/hydra dependency is needed.
"""
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from . import maskops
from .device import DEVICE, autocast, clear_cache

CACHE = Path(__file__).resolve().parent.parent / "models"
os.environ.setdefault("HF_MODULES_CACHE", str(CACHE / "modules"))  # remote code lives with the weights

# Pinned commits: remote code is executed, so never pull a newer revision silently.
REPOS = {
    "BiRefNet": ("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"),
    "BiRefNet-matting": ("ZhengPeng7/BiRefNet-matting", "eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9"),
    "SAM 2.1 small": ("facebook/sam2.1-hiera-small", "ee5bba1d82bb8749febdf90f45e84b687142ba03"),
}
DOWNLOAD_MB = {"BiRefNet": 445, "BiRefNet-matting": 885, "SAM 2.1 small": 185}
BIREFNET_RES = 1024
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class ModelMissing(Exception):
    """Model not in models/ and download not allowed. The message is the REPOS key."""


def is_cached(name):
    repo, rev = REPOS[name]
    return (CACHE / f"models--{repo.replace('/', '--')}" / "snapshots" / rev).exists()


def _load(cls, name, on_status=lambda msg: None, download=False, **kw):
    repo, rev = REPOS[name]
    try:  # offline first: no network once the model is cached
        return cls.from_pretrained(repo, revision=rev, cache_dir=CACHE, local_files_only=True, **kw)
    except OSError:
        if not download:  # never fetch without the user saying yes
            raise ModelMissing(name)
        on_status(f"Downloading {name} (~{DOWNLOAD_MB[name]} MB, first run only, can take minutes)…")
        return cls.from_pretrained(repo, revision=rev, cache_dir=CACHE, **kw)


def _load_model(cls, name, on_status=lambda msg: None, **kw):
    # fp32 weights everywhere (BiRefNet ships fp16); CUDA gets fp16 compute via autocast
    return _load(cls, name, on_status, dtype=torch.float32, **kw)


def subject_mask(rgb, raw, reach=0.0):
    """Full-res uint8 mask from Engine.subject_raw output. reach (logits) > 0 takes in more. Needs no model."""
    out = np.zeros(rgb.shape[:2], np.uint8)
    if raw is None:
        return out
    p, (x0, y0, x1, y1), region = raw
    m = maskops.guided_upsample(maskops.shift_confidence(p, reach), rgb[y0:y1, x0:x1], BIREFNET_RES)
    out[y0:y1, x0:x1] = m if region is None else np.minimum(m, region)
    return out


def _spread(mask, n):
    """Up to n (x, y) points of a boolean mask, spread out by farthest-point sampling."""
    ys, xs = np.nonzero(mask)
    p = np.stack([xs, ys], 1).astype(np.float32)[::max(1, len(xs) // 4096)]
    if not len(p):
        return []
    pick = [p[len(p) // 2]]
    d = ((p - pick[0]) ** 2).sum(1)
    while len(pick) < min(n, len(p)) and d.max() > 0:
        pick.append(p[d.argmax()])
        d = np.minimum(d, ((p - pick[-1]) ** 2).sum(1))
    return [(float(a), float(b)) for a, b in pick]


class Engine:
    def __init__(self):
        self.name = None
        self.model = None
        self.processor = None
        self.sam_cache = OrderedDict()  # image id -> embeddings (CPU while SAM is unloaded)

    # ---------- model switching ----------

    def use(self, name, on_status=lambda msg: None, download=False):
        if self.name == name:
            return
        self.unload()
        on_status(f"Loading {name}…")
        try:
            if name == "SAM 2.1 small":
                from transformers import Sam2Model, Sam2Processor
                self.processor = _load(Sam2Processor, name, on_status, download)
                model = _load_model(Sam2Model, name, on_status, download=download)
            else:
                from transformers import AutoModelForImageSegmentation
                model = _load_model(AutoModelForImageSegmentation, name, on_status, download=download,
                                    trust_remote_code=True)
        except ModelMissing:  # the window offers the download
            raise
        except Exception as e:  # the window shows load failures in a dialog: the user has to act on them
            raise RuntimeError(f"Could not load {name}: {e}") from e
        self.model = model.to(DEVICE).eval()
        self.name = name
        if name == "SAM 2.1 small":
            for k, emb in self.sam_cache.items():
                self.sam_cache[k] = [e.to(DEVICE) for e in emb]

    def download(self, name, on_status=lambda msg: None):
        self.use(name, on_status, download=True)

    def unload(self):
        if self.model is None:
            return
        if self.name == "SAM 2.1 small":
            for k, emb in self.sam_cache.items():
                self.sam_cache[k] = [e.cpu() for e in emb]
        self.model = self.processor = self.name = None
        clear_cache()

    # ---------- BiRefNet ----------

    @torch.inference_mode()
    def _birefnet(self, rgb):
        """Soft mask (float 0..1) at model resolution for an RGB crop."""
        x = cv2.resize(rgb, (BIREFNET_RES, BIREFNET_RES),
                       interpolation=cv2.INTER_AREA if max(rgb.shape[:2]) > BIREFNET_RES else cv2.INTER_CUBIC)
        x = (x.astype(np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        with autocast():
            out = self.model(x)[-1]
        return out.float().sigmoid()[0, 0].cpu().numpy()

    def subject(self, rgb, rect=None, region=None, on_status=lambda msg: None):
        """Foreground mask of the whole image, inside rect=(x0, y0, x1, y1), or inside a painted region."""
        return subject_mask(rgb, self.subject_raw(rgb, rect, region, on_status))

    def paint(self, image_id, rgb, paint, on_status=lambda msg: None):
        """The things inside the paint (uint8 mask at any res): SAM prompted from the paint, clipped to it plus reach px.

        Points deep inside the paint include, points away from it exclude, so background painted over at the
        rough edge is trimmed; SAM sees the whole image for context.
        """
        H, W = rgb.shape[:2]
        reach = max(4, round(0.02 * max(H, W)))  # same rule as refine
        sx, sy = W / paint.shape[1], H / paint.shape[0]
        # dilate at paint res: a ~2*reach px kernel at full res is slow
        r = max(1, round(reach / sx))
        region = cv2.resize(maskops.grow(paint, r), (W, H), interpolation=cv2.INTER_LINEAR)
        on = (paint > 127).astype(np.uint8)
        din = cv2.distanceTransform(on, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        m = din.max()
        pos = _spread(din >= min(0.5 * m, max(8, 0.25 * m)), 8)  # deep inside: never on a rough stroke edge
        x, y, w, h = cv2.boundingRect(on)
        near = np.zeros_like(on)  # negatives describe the nearby background, not far corners
        near[max(0, y - h // 4):y + h + h // 4, max(0, x - w // 4):x + w + w // 4] = 1
        dout = cv2.distanceTransform(1 - on, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        neg = _spread((dout >= 2 * r) & (near > 0), 8)
        pts = [((px + 0.5) * sx, (py + 0.5) * sy) for px, py in pos + neg]
        box = (max(0, (x - r) * sx), max(0, (y - r) * sy), min(W, (x + w + r) * sx), min(H, (y + h + r) * sy))
        out = np.minimum(self.sam(image_id, rgb, pts, [1] * len(pos) + [0] * len(neg), box, on_status), region)
        # objects the deep points missed (a castle on a painted hill): keep each one that lies inside the paint;
        # the widest mask at a point, so a field of the painted-over valley is judged as the whole valley
        inside = region > 127
        rest = maskops.grow(on & (cv2.resize(out, paint.shape[::-1], interpolation=cv2.INTER_AREA) <= 127), -r)
        for px, py in _spread(rest, 16):  # ponytail: one SAM call per point, batch the prompts if this is slow
            m = self.sam(image_id, rgb, [((px + 0.5) * sx, (py + 0.5) * sy)], [1], None, on_status, True)
            k = m > 127
            if k.any() and (k & ~inside).sum() < 0.02 * k.sum():
                out = np.maximum(out, np.minimum(m, region))
        if not (out > 127).any():  # SAM found nothing: keep the paint, border snapped to the photo
            out = maskops.guided_upsample(paint.astype(np.float32) / 255, rgb, max(paint.shape))
        return out

    def subject_raw(self, rgb, rect=None, region=None, on_status=lambda msg: None):
        """Model output for subject_mask: (probability map at model res, crop box, region crop or None).

        region: uint8 HxW, 255 = area of interest. Only its padded bounding box is processed, and the result
        is clipped to the region. rect is treated as a filled rectangular region.
        """
        self.use("BiRefNet", on_status)
        H, W = rgb.shape[:2]
        if region is None and rect is not None:
            x0, y0, x1, y1 = rect
            region = np.zeros((H, W), np.uint8)
            region[y0:y1, x0:x1] = 255
        if region is not None:
            rect = maskops.bbox(region, 127)
            if rect is None:
                return None
            # pad for context, then widen the short side to at most 3:1 so the square resize doesn't squash it
            x0, y0, x1, y1 = maskops.pad_box(rect, round(0.1 * max(rect[2] - rect[0], rect[3] - rect[1])), W, H)
            nw = min(W, max(x1 - x0, -(-(y1 - y0) // 3)))
            nh = min(H, max(y1 - y0, -(-(x1 - x0) // 3)))
            x0, y0 = min(max(0, (x0 + x1 - nw) // 2), W - nw), min(max(0, (y0 + y1 - nh) // 2), H - nh)
            rect = (x0, y0, x0 + nw, y0 + nh)
        x0, y0, x1, y1 = rect or (0, 0, W, H)
        on_status("Detecting subject…")
        # no grey fill outside the region: the model then mattes the whole non-grey blob, background included
        return self._birefnet(rgb[y0:y1, x0:x1]), (x0, y0, x1, y1), None if region is None else region[y0:y1, x0:x1]

    def refine(self, rgb, mask, on_status=lambda msg: None):
        """Re-matte the masked region with BiRefNet-matting for soft hair/branch alpha."""
        box = maskops.bbox(mask)
        if box is None:
            return mask
        self.use("BiRefNet-matting", on_status)
        H, W = mask.shape
        bw, bh = box[2] - box[0], box[3] - box[1]
        x0, y0, x1, y1 = maskops.pad_box(box, max(16, int(0.08 * max(bw, bh))), W, H)
        crop, m = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        on_status("Refining edges…")
        reach = max(4, round(0.02 * max(x1 - x0, y1 - y0)))
        outer, core = maskops.band(m, reach)
        # no grey fill beyond the band: the model then mattes the whole band ring as foreground
        matte = maskops.guided_upsample(self._birefnet(crop), crop, BIREFNET_RES, eps=1e-4)
        out = mask.copy()
        # the matte only decides inside the band: it may add detail up to reach px out, never remove the core
        out[y0:y1, x0:x1] = np.maximum(np.minimum(matte, outer), core)
        return out

    # ---------- SAM 2.1 ----------

    SAM_IN = 1024

    def _sam_inputs(self, rgb):
        H, W = rgb.shape[:2]
        s = min(1.0, self.SAM_IN / max(H, W))
        small = rgb if s == 1.0 else cv2.resize(rgb, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        return small, s

    @torch.inference_mode()
    def sam_prepare(self, image_id, rgb, on_status=lambda msg: None):
        """Load SAM and cache this image's embedding so the first click is fast."""
        self.use("SAM 2.1 small", on_status)
        small, _ = self._sam_inputs(rgb)
        emb = self.sam_cache.get(image_id)
        if emb is None:
            on_status("Encoding image for SAM…")
            pix = self.processor(images=small, return_tensors="pt")["pixel_values"].to(DEVICE)
            with autocast():
                emb = self.model.get_image_embeddings(pix)
            self.sam_cache[image_id] = emb
            while len(self.sam_cache) > 4:
                self.sam_cache.popitem(last=False)
        self.sam_cache.move_to_end(image_id)
        return emb

    @torch.inference_mode()
    def sam(self, image_id, rgb, points, labels, box=None, on_status=lambda msg: None, largest=False):
        """points: [(x, y)] in image px, labels: 1 include / 0 exclude, box: (x0, y0, x1, y1) or None.

        largest: for a single point, the widest of SAM's 3 masks instead of the most confident one.
        """
        emb = self.sam_prepare(image_id, rgb, on_status)
        small, s = self._sam_inputs(rgb)
        size = [[small.shape[0], small.shape[1]]]
        kw = {}
        if points:
            kw["input_points"] = [[[[x * s, y * s] for x, y in points]]]
            kw["input_labels"] = [[list(labels)]]
        if box is not None:
            kw["input_boxes"] = [[[v * s for v in box]]]
        enc = self.processor(original_sizes=size, return_tensors="pt", **kw)
        args = {k: enc[k].to(DEVICE) for k in ("input_points", "input_labels", "input_boxes") if k in enc}
        single = len(points) == 1 and box is None  # ambiguous prompt: let SAM pick among 3 masks
        with autocast():
            out = self.model(image_embeddings=emb, multimask_output=single, **args)
        logits = out.pred_masks[0, 0].float()  # (n_masks, h, w), covers the whole (resized) image
        best = int((logits > 0).sum((1, 2)).argmax() if largest else out.iou_scores[0, 0].argmax()) if single else 0
        H, W = rgb.shape[:2]
        k = min(1.0, maskops.WORK / max(H, W))
        # upsample logits (not probabilities) so the edge stays sharp before the guided filter
        lg = torch.nn.functional.interpolate(logits[best][None, None], size=(round(H * k), round(W * k)), mode="bilinear")
        # SAM logits are uncalibrated (threshold 0); sharpen so the interior is solid and the
        # guided filter, not logit noise, decides the soft edge
        soft = (lg * 4).sigmoid()[0, 0].cpu().numpy()
        return maskops.guided_upsample(soft, rgb, logits.shape[-1])
