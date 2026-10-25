# CutOut

Cut subjects out of images on your own machine. Open an image, select what to keep, check the live preview and export a transparent PNG.

Everything runs locally. The first time a feature needs an AI model, CutOut asks before downloading it and shows its size. After that it works offline.

## Install and run

You need [uv](https://docs.astral.sh/uv/) and either a Mac with Apple Silicon or a Windows PC with an NVIDIA GPU (driver 580+). Other machines fall back to the CPU, which is slower.

```bash
git clone https://github.com/MakiNoAtorie/Maki-s-CutOut.git
cd Maki-s-CutOut
uv run main.py
```

The first run sets up the environment. You can also pass an image: `uv run main.py photo.jpg`.

**Mac app:** `scripts/make_mac_app.sh` puts `CutOut.app` in `/Applications`. The app runs this folder, so run the script again if you move the folder.

**Windows shortcut:** `powershell -ExecutionPolicy Bypass -File scripts\make_windows_shortcut.ps1` adds CutOut to the Start menu. It opens without a console window and logs to `%LOCALAPPDATA%\CutOut\CutOut.log`. Run the script again if you move the folder or delete `.venv`.

## How to use

1. **Open** an image: use Open, paste it (⌘V / Ctrl+V), or drop it on the window.
2. **Select** the part to keep, using one of four modes (keys 1–4):

   | Mode | What it does |
   |---|---|
   | Auto subject | Finds the main subject as soon as the image loads. Drag a rectangle to limit the search to one area. |
   | Paint region | Paint roughly over what you want and it is cut out cleanly. Right-drag or Alt-drag erases. |
   | Click select | Left-click to include, right-click to exclude, drag to draw a box. |
   | Color key | Click a colour, for example the sky, to make it transparent. Instant, no model needed. |

3. **Apply** the selection with Enter. **Combine** decides how each new selection joins the current one: Replace, Add, Subtract or Intersect.
4. **Polish** the edges:
   - **Refine edges** gives finer edges on hair and branches.
   - **Feather** softens the edge.
   - **Shrink/Grow** moves the edge in or out.
   - **Defringe** removes background colour bleeding into the edge.
5. **Export** a full-resolution PNG, optionally cropped to the subject. You can also save only the mask, or copy the result to the clipboard. Your original file is never changed.

**Undo** with ⌘Z or Ctrl+Z, **redo** with ⇧⌘Z or Ctrl+Shift+Z. **Esc** clears the clicks or paint you haven't applied yet. Zoom with the scroll wheel or a pinch, pan with Space+drag. ⌘0 / Ctrl+0 fits the image to the window, ⌘1 / Ctrl+1 shows it at 100%. `[` and `]` change the brush size.

## Credits

CutOut is a front end for these models:

- **[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)** by Peng Zheng et al. ([paper](https://arxiv.org/abs/2401.03407), MIT). Used for subject detection, Paint region and Refine edges.
- **[SAM 2.1](https://github.com/facebookresearch/sam2)** by Meta AI ([paper](https://arxiv.org/abs/2408.00714), Apache 2.0). Used for Click select.

It is built with [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://github.com/huggingface/transformers), [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/) (LGPL), [OpenCV](https://opencv.org/), [kornia](https://github.com/kornia/kornia), [timm](https://github.com/huggingface/pytorch-image-models) and [Pillow](https://python-pillow.org/).

Model weights are downloaded from Hugging Face and keep their own licenses.

## License

[MIT](LICENSE). This covers CutOut's own code; the models and libraries above keep their own licenses.
