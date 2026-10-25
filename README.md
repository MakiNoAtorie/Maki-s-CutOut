# CutOut

Local image extraction: drop in an image, pick what to keep, watch the live preview, export a PNG with alpha.
Nothing is downloaded without asking: the first time a mode needs a model, the app offers it with its size (into `models/`).
After that it runs fully offline.

## Launch

Needs [uv](https://docs.astral.sh/uv/). The first run creates the environment.

macOS (Apple Silicon, MPS):

```bash
uv run main.py
```

Windows (NVIDIA, CUDA):

```bash
uv run main.py
```

Optionally pass an image: `uv run main.py photo.jpg`.

macOS app: `scripts/make_mac_app.sh` builds `/Applications/CutOut.app`, a launcher for this folder (re-run it if the folder moves). Log: `~/Library/Logs/CutOut.log`.

Windows pulls PyTorch from the CUDA 13.0 index (Blackwell / RTX 50xx needs CUDA 12.8+); this requires an NVIDIA driver 580 or newer.
The device is picked automatically (CUDA → MPS → CPU) and shown in the status bar with the loaded model.

## Modes

| Mode | How |
|---|---|
| Auto subject | BiRefNet finds the main subject. Runs on load; drag a rectangle to limit it to an area. |
| Paint region | Paint over the area you want (right-drag or Alt-drag erases). BiRefNet runs only there, so a tree in a landscape can be pulled out on its own. |
| Click select | SAM 2.1: left click include, right click exclude, drag for a box. |
| Color key | Click a colour (e.g. sky) to make it transparent. Tolerance, softness, invert. No model, instant. |

Models per mode: Auto/Paint → BiRefNet (445 MB), Refine edges → BiRefNet-matting (885 MB), Click select → SAM 2.1 small (185 MB), Color key → none.

**Combine** (Replace / Add / Subtract / Intersect) sets how the next selection merges with the current mask.
Prompt-based modes show a live candidate; press **Apply** (Enter) to commit it. Undo (Cmd+Z, or Ctrl+Z on Mac) steps back one prompt, then through committed masks, Refine included; Esc clears all prompts.

**Refine edges** re-mattes the committed mask (one undoable step) with BiRefNet-matting for hair and branches. It can add detail near the edge
but only trims a little inside it, and works best on Auto, Paint and Click masks.
**Feather**, **Shrink/Grow** and **Defringe** (removes background colour bleed from edge pixels) apply live.

Navigation: mouse wheel, Ctrl/Cmd+scroll or trackpad pinch zooms; trackpad scroll, middle-drag or Space+drag pans.
Views are linked by default (toggle above the preview). Keys: 1–4 switch mode, Ctrl+0 fit, Ctrl+1 100% (Cmd on Mac), [ / ] brush size.

## Export

Export → PNG at full resolution (default), optionally cropped to the mask bounds; save the mask alone as a grayscale PNG;
or copy the result to the clipboard. The source file is never written.

## Layout

```
main.py            entry point
app/               GUI (window.py: toolbar, modes, workers; viewer.py: zoom/pan view with proxy rendering)
engine/            device.py (device + fp16 policy), models.py (BiRefNet, SAM 2.1, one model loaded at a time),
                   maskops.py (guided upsampling, colour key, combine, feather/grow/defringe)
scripts/           make_test_images.py (synthetic test images incl. a 50 MP one), smoke_test.py (headless check)
```

Checks: `uv run python engine/maskops.py` (no models) and `uv run python scripts/smoke_test.py IMAGE` (loads all three models).
