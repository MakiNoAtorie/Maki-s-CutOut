# CutOut — debug pass (2026-10-25)

Scope: `main.py`, `app/window.py`, `app/viewer.py`, `engine/*.py`, `scripts/*.py`, `pyproject.toml`, `README.md`.
Method: full read, then headless checks (no model was loaded): `engine/maskops.py` self-check, offscreen `MainWindow`
with a stubbed engine for the race conditions, unit checks for the numeric claims. "Confirmed" = reproduced in a check;
"by reading" = traced in code only. P1 = crash or wrong result on a normal path, P2 = wrong result in realistic cases,
P3 = edge/polish.

## P1

### 1. Copy to clipboard with "Crop to mask bounds" crashes — confirmed
`app/window.py:652-655` crops `_latest` with a slice, `app/window.py:689` passes that non-contiguous view to
`to_qimage`, which builds a `memoryview` → `BufferError: memoryview: underlying buffer is not C-contiguous`.
The exception fires inside the `done` slot, so nothing is copied and the user gets a traceback on stderr only.
Export is unaffected because `cv2` accepts strided arrays.
**Fix:** in `_final`, wrap the cropped arrays in `np.ascontiguousarray(...)` (both `a` and `rgba`). One place, fixes
copy and keeps export as is.

### 2. Undo during an in-flight SAM/paint job leaves a ghost candidate — confirmed
`app/window.py:492-496` (`run_sam`) and `app/window.py:453-455` (`run_paint`): when no prompts remain they set
`candidate = None` and refresh, but the job started by the previous prompt is still running on the worker. Its result
lands in `_set_candidate`, so a candidate with no visible prompt appears; Enter commits it. Repro: click once in
Click select, Ctrl+Z before SAM finishes (always on the first click, which loads the model), wait.
**Fix:** call `self.models.cancel("sam")` in the `run_sam` early-return branch and `self.models.cancel("paint")` in
the `run_paint` early-return branch (the same thing `undo()` already does for the box key at line 569).

### 3. Late initial detection wipes a candidate made in another mode — confirmed
`app/window.py:537-541` `commit()` sets `self.candidate = None`. The whole-image detection started on load
(`_subject_done`) can land long after the user has switched to Color key and picked a colour (the key job runs on the
`cpu` worker, so it finishes first). The pick overlay stays, the candidate disappears, Apply is disabled. Same race
for any prompt mode if the user is faster than the first model download. Repro in the stub run: pick a colour while
a slow subject job runs → `key_pick` set, `candidate None`.
**Fix:** remove `self.candidate = None` from `commit()`. `apply()` calls `reset_session()` right after `commit()`,
which clears the candidate anyway, so `apply` keeps its behaviour and `_subject_done` stops clobbering prompts.

### 4. A failed Refine job disables Refine silently until the mask changes — confirmed
`app/window.py:604-608`: `_refining = base` is set before the job runs and only ever compared, never reset on failure.
If `engine.refine` raises (model download failed, OOM), the dialog shows once, `refresh()` sees `_refining is base`
forever, and the unrefined mask is rendered while the Refine button stays checked. Toggling Refine off and on does
not retry (stub run: 1 attempt before and after toggling).
**Fix:** in `_failed`, add `self._refining = None`. Harmless for other failures; next `refresh()` with Refine on
retries.

## P2

### 5. Defringe colour estimate is halved at mask boundaries on images over ~1.5k px — confirmed (unit level)
`engine/maskops.py:158`: `_spread` downsamples the 0/1 weight map `w` as **uint8** with `INTER_AREA`, so fractional
coverage rounds to 0 or 1 while the numerator (`rgb * w`) keeps its true average. At a low-res cell half covered by
foreground, `den` is 1.0 instead of 0.5 and `num/den` returns half the real colour (check: 100 vs 200 on a flat
200-grey image). Only when `f > 1` (`defringe` line 184: long side ≥ 1536 px); at `f = 1` the resize is a no-op,
which is why the `__main__` self-check (400×300) passes. End-to-end on a wide soft edge the error averages out
(f=2 disc test still recovered red 219.9/220); it bites on hard-edged masks (SAM, Color key with low softness) where
the boundary cells *are* the estimate.
**Fix:** `den = cv2.resize(w.astype(np.float32), size, interpolation=cv2.INTER_AREA)`. Add an `f = 2` case to the
`__main__` self-check (e.g. 2000×1600 disc) so it stays caught.

### 6. Defringe pulls edge pixels toward black where no solid foreground is nearby — by reading
`engine/maskops.py:161-168`: `_spread` fills holes with three Gaussian passes up to `sigma * 16` (~80 low-res px);
pixels still uncovered keep `out = num = 0`, i.e. black. `defringe` line 195 then uses that black `F` for every edge
pixel with `a < 0.25`. Reachable with wide semi-transparent masks: Color key with high softness over a gradient sky,
a 50 px feather, or Shrink/Grow pulling the solid core far from the edge.
**Fix:** after the loop in `_spread`, fill what is left with the global weighted mean colour:
`rest = have < 1e-3; out[rest] = num.sum((0, 1)) / max(den.sum(), 1e-3)`. One line; no more black.

### 7. Painted-region detection feeds the model a squashed, context-free crop — by reading
`engine/models.py:106-115`: the crop is the exact bbox of the painted blob with no padding, then stretched to
1024×1024. A stroke along a tree trunk (say 300×2000 px) becomes a 1:7 distortion; BiRefNet was not trained on that.
`refine()` (line 132) already pads its box by 8% for the same reason.
**Fix:** after `rect = maskops.bbox(region, 127)`, pad it: 10% of the long side on every side, and extend the short
side so the aspect ratio is at most 3:1 (keep inside the image with `pad_box`/clamping). The result is still clipped
by `np.minimum(m, region)` at line 119, so nothing outside the paint leaks into the mask.

### 8. Small crops are upscaled with INTER_AREA (nearest-neighbour-like) before BiRefNet — by reading
`engine/models.py:91`: `cv2.resize(rgb, (1024, 1024), INTER_AREA)`. INTER_AREA is correct for shrinking but behaves
like nearest when enlarging, so a 400 px box or paint blob reaches the model as 2.5× blocky pixels; edges come out
staircased before the guided filter.
**Fix:** `interpolation=cv2.INTER_AREA if max(rgb.shape[:2]) > BIREFNET_RES else cv2.INTER_CUBIC`.

### 9. 16-bit PNG/TIFF loads as a near-white image — by reading
`app/window.py:70-72`: `Image.convert("RGB")` on Pillow modes `I;16`, `I;16B`, `I;16L`, `I` clips values above 255 to
white instead of scaling. `IMAGE_FILTER` advertises `.tif/.tiff` and 16-bit PNGs are common from scanners and
phone RAW pipelines.
**Fix:** in `load_image`, before `convert`, if `im.mode` is `I` or starts with `I;16`: `a = np.asarray(im)`,
`a = (a >> 8).astype(np.uint8)` for 16-bit (`np.clip(a // 257, 0, 255)` for `I`), then `np.dstack([a] * 3)`; keep
`exif_transpose` first. Verify with a synthetic `I;16` PNG written by Pillow.

### 10. Mask history is unbounded in bytes: up to 1.5 GB on a 50 MP image — by reading
`app/window.py:24, 538`: 30 full-resolution uint8 masks × 50 MB each, plus the redo stack holding what was popped.
The README ships a 50 MP stress image; two apply/undo rounds of Combine experiments get there.
**Fix:** cap by bytes, not count: `limit = max(3, 400_000_000 // mask.nbytes)` in `commit()` and slice
`self.history[-limit:]`. Keep `HISTORY = 30` as the upper bound for small images.

## P3

### 11. Pasting when the clipboard reports an image but delivers a null one throws — confirmed
`app/window.py:75-79`: `qimage_to_rgb(QImage())` → `TypeError: a bytes-like object is required, not 'NoneType'`
(`constBits()` is None). Some apps set the image MIME type without data.
**Fix:** `if img.isNull(): return None` in `qimage_to_rgb`; `paste`/`dropEvent` show "Clipboard has no image" on None.

### 12. Export filename loses a dotted stem
`app/window.py:672`: `Path(p).with_suffix(".png")` turns `photo.v2` into `photo.png`.
**Fix:** `if p.suffix.lower() != ".png": p = p.with_name(p.name + ".png")`.

### 13. Colour-key distance cache: 200 MB kept after image change, keyed on the worker thread
`app/window.py:509-510`: `self.image_id` is read inside the cpu job and the float32 map is never dropped on load.
**Fix:** capture `iid = self.image_id` next to `pick` on the GUI thread and compare against it; add
`self._dist = (None, None)` to `set_image`.

### 14. Box-limited detect that finds nothing shows the pending line, not the "No clear subject" hint
`app/window.py:487-490` only hints on the whole-image path (noted as a leftover in UX_REVIEW).
**Fix:** in `_set_candidate`, if `not (m > 127).any()`, show the hint for 8 s instead of `PENDING`.

### 15. `.gitignore` misses the real test folder
`Test images/` (with a space, five Unsplash JPGs) is not ignored while `test_images/` is. `git init` would stage ~20 MB.
**Fix:** add `Test images/`.

### Documented, no fix now
- `app/window.py:634-637` `_show_proxy` swaps `preview.proxy` but leaves `preview.full` on the previous render; a
  zoom during the proxy→full gap briefly shows the stale full-res. Cosmetic.
- `app/viewer.py:237-239` `enterEvent` → `setFocus()` steals focus from combos/sliders on hover (needed for Space-pan).
- `app/window.py:437-448` `on_paint` rebuilds a proxy-size RGBA pixmap on every mouse move (16 MB at 2048 px). Not
  measured; throttle with a 16 ms timer if the brush feels laggy.
- Model download has no progress (UX_REVIEW finding 6, status text only).

## Checked and fine
- `_latest` consistency: both runners are single-thread FIFO, so export/copy always see the newest render's result.
- Shrink/Grow and Feather at full res: 101 px ellipse dilate on 12 MP is 0.17 s (≈1 s at 50 MP); not a bottleneck.
- Shortcut forms (`setShortcuts(["Return","Enter"])`, `Qt.Key_Escape`, `QKeySequence.Open`) all resolve.
- SAM 2 path: no padding in Sam2 preprocessing, so stretching the 256² logits back to the image aspect is correct;
  `original_sizes` without images is supported by `Sam2Processor`.
- `guided_upsample`, `combine`, `bbox`, `key_alpha` as per the `__main__` self-check (passes).
- `_guard`/stale handling across image changes; `closeEvent` + `os._exit` shutdown.

## Verification for the fix pass (no model loads — do not run `scripts/smoke_test.py`)
1. `uv run python engine/maskops.py` passes, including the new `f = 2` defringe case.
2. Offscreen (`QT_QPA_PLATFORM=offscreen`) `MainWindow` with a stubbed engine: (2) undo during in-flight SAM leaves
   `candidate None`; (3) colour pick during a slow subject job keeps its candidate after the commit; (4) a raising
   `refine` is retried after toggling Refine off/on; (1) `copy_result` with crop checked produces a QImage.
   Patch `QMessageBox.warning` to a no-op in the test — it is modal and hangs offscreen.
3. `load_image` on a Pillow-written `I;16` PNG returns a non-white uint8 RGB.
4. `qimage_to_rgb(QImage())` returns None without raising.
