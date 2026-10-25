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
        box = (max(0, (x - r) * sx), max(0, (y - r) * sy), min(W, (x + w + r) * sx), min(H, (y + h + r) * sy))
        # drop an include point whose own SAM object has (in paint) / (in box) < 0.8: dome valley 0.48-0.70, forest 0.87-1.00
        emb = self.sam_prepare(image_id, rgb, on_status)
        small, s = self._sam_inputs(rgb)
        b = np.s_[max(0, y - r):y + h + r, max(0, x - r):x + w + r]  # box at paint res
        keep = []
        for px, py in pos:
            o = self._sam_decode(emb, small.shape[:2], [((px + 0.5) * sx * s, (py + 0.5) * sy * s)], [1], None, True)
            o = torch.nn.functional.interpolate(o[None, None], size=paint.shape, mode="bilinear")[0, 0].cpu().numpy()[b] > 0
            if (o & on[b].astype(bool)).sum() >= 0.8 * o.sum():
                keep.append((px, py))
        pos = keep or [tuple(np.unravel_index(din.argmax(), din.shape)[::-1])]
        pts = [((px + 0.5) * sx, (py + 0.5) * sy) for px, py in pos + neg]
        out = np.minimum(self.sam(image_id, rgb, pts, [1] * len(pos) + [0] * len(neg), box, on_status), region)
        # what SAM left out of the paint (a castle on a painted hill): BiRefNet on each leftover piece, in context
        left = on & (cv2.resize(out, paint.shape[::-1], interpolation=cv2.INTER_AREA) <= 127)
        left = cv2.morphologyEx(left, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop specks and threads
        n, lab, st, _ = cv2.connectedComponentsWithStats(left)
        for i in sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:8]:
            if st[i, cv2.CC_STAT_AREA] < (0.01 * max(paint.shape)) ** 2:
                break
            piece = cv2.resize(maskops.grow((lab == i).astype(np.uint8) * 255, r), (W, H), interpolation=cv2.INTER_LINEAR)
            m = self.subject(rgb, region=piece, on_status=on_status)
            if (m > 127)[piece > 127].mean() < 0.95:  # nearly all the piece: BiRefNet took sky or valley as the subject
                out = np.maximum(out, m)
        inside = region > 127
        out = maskops.fill_holes(out, ~inside)  # SAM speckle inside the paint: background must reach outside it
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
        if (matte > 127)[outer > 127].mean() < 0.02:  # no subject for the matting model (a whole hill): keep the mask
            on_status("Refine found no subject edges in this selection")
            return mask
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
    def _sam_embed(self, rgb):
        pix = self.processor(images=self._sam_inputs(rgb)[0], return_tensors="pt")["pixel_values"].to(DEVICE)
        with autocast():
            return self.model.get_image_embeddings(pix)

    @torch.inference_mode()
    def _sam_decode(self, emb, size, points, labels, box, multimask, masks=None):
        """Low-res logits (h, w) of SAM's mask, the most confident one if multimask.

        Prompts in px of an image of size (H, W). masks: (1, 1, h, w) logits of an earlier pass as a mask prompt, or None.
        """
        kw = {}
        if points:
            kw["input_points"] = [[[list(p) for p in points]]]
            kw["input_labels"] = [[list(labels)]]
        if box is not None:
            kw["input_boxes"] = [[list(box)]]
        enc = self.processor(original_sizes=[list(size)], return_tensors="pt", **kw)
        args = {k: enc[k].to(DEVICE) for k in ("input_points", "input_labels", "input_boxes") if k in enc}
        with autocast():
            out = self.model(image_embeddings=emb, multimask_output=multimask, input_masks=masks, **args)
        return out.pred_masks[0, 0, int(out.iou_scores[0, 0].argmax()) if multimask else 0].float()

    @torch.inference_mode()
    def sam_prepare(self, image_id, rgb, on_status=lambda msg: None):
        """Load SAM and cache this image's embedding so the first click is fast."""
        self.use("SAM 2.1 small", on_status)
        emb = self.sam_cache.get(image_id)
        if emb is None:
            on_status("Encoding image for SAM…")
            emb = self.sam_cache[image_id] = self._sam_embed(rgb)
            while len(self.sam_cache) > 4:
                self.sam_cache.popitem(last=False)
        self.sam_cache.move_to_end(image_id)
        return emb

    @torch.inference_mode()
    def sam(self, image_id, rgb, points, labels, box=None, on_status=lambda msg: None):
        """points: [(x, y)] in image px, labels: 1 include / 0 exclude, box: (x0, y0, x1, y1) or None.

        A second pass prompts SAM again on the first mask's box, padded 10%: one logit of the whole image is
        ~23 px of a 5853 px photo, too coarse for the guided filter to find the edge.
        """
        emb = self.sam_prepare(image_id, rgb, on_status)
        small, s = self._sam_inputs(rgb)
        single = len(points) == 1 and box is None  # ambiguous prompt: let SAM pick among 3 masks
        logits = self._sam_decode(emb, small.shape[:2], [(x * s, y * s) for x, y in points], labels,
                                  None if box is None else [v * s for v in box], single)  # covers the whole (resized) image
        H, W = rgb.shape[:2]
        out = np.zeros((H, W), np.uint8)
        x0, y0, x1, y1 = 0, 0, W, H
        lh, lw = logits.shape
        b = maskops.bbox((logits > 0).cpu().numpy(), 0)
        if b is not None and (b[2] - b[0]) * (b[3] - b[1]) <= 0.6 * lh * lw:  # most of the image: no gain
            b = (b[0] * W // lw, b[1] * H // lh, -(-b[2] * W // lw), -(-b[3] * H // lh))
            x0, y0, x1, y1 = maskops.pad_box(b, round(0.1 * max(b[2] - b[0], b[3] - b[1])), W, H)
            # first-pass logits resampled onto the crop: the mask prompt
            th = torch.tensor([[(x1 - x0) / W, 0, (x0 + x1) / W - 1], [0, (y1 - y0) / H, (y0 + y1) / H - 1]],
                              device=logits.device)
            grid = torch.nn.functional.affine_grid(th[None], [1, 1, lh, lw], align_corners=False)
            prior = torch.nn.functional.grid_sample(logits[None, None], grid, align_corners=False)
            rgb = rgb[y0:y1, x0:x1]
            small, s = self._sam_inputs(rgb)
            # only prompts inside the crop; with none left the prior alone carries the first pass
            inside = [i for i, (x, y) in enumerate(points) if x0 <= x < x1 and y0 <= y < y1]
            pts = [((points[i][0] - x0) * s, (points[i][1] - y0) * s) for i in inside]
            crop_box = None
            if box is not None:
                cb = [min(max(v - o, 0), e) for v, o, e in zip(box, (x0, y0, x0, y0), (x1 - x0, y1 - y0) * 2)]
                crop_box = [v * s for v in cb] if cb[2] > cb[0] and cb[3] > cb[1] else None
            logits = self._sam_decode(self._sam_embed(rgb), small.shape[:2], pts, [labels[i] for i in inside],
                                      crop_box, False, prior)
            H, W = rgb.shape[:2]
        k = min(1.0, maskops.WORK / max(H, W))
        # upsample logits (not probabilities) so the edge stays sharp before the guided filter
        lg = torch.nn.functional.interpolate(logits[None, None], size=(round(H * k), round(W * k)), mode="bilinear")
        # SAM logits are uncalibrated (threshold 0); sharpen so the interior is solid and the
        # guided filter, not logit noise, decides the soft edge
        soft = (lg * 4).sigmoid()[0, 0].cpu().numpy()
        out[y0:y1, x0:x1] = maskops.guided_upsample(soft, rgb, logits.shape[-1])  # first logits < 0 outside
        return out
