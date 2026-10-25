import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QMimeData, QObject, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication, QImage, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
    QProgressBar, QPushButton, QSlider, QSplitter, QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

from engine import maskops
from engine.device import device_label
from engine.models import DOWNLOAD_MB, Engine, is_cached, subject_mask

from .viewer import ImageView, make_proxy, to_qimage

Image.MAX_IMAGE_PIXELS = 300_000_000
MODES = ["Auto subject", "Paint region", "Click select", "Color key"]
OPS = ["Replace", "Add", "Subtract", "Intersect"]
HISTORY = 30
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
PENDING = "Selection pending — Enter applies, Esc discards"


class Runner(QObject):
    """One worker thread. A newer submit with the same key makes older ones stale (skipped / dropped)."""
    done = Signal(object, object)
    failed = Signal(str)
    status = Signal(str)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.pool = ThreadPoolExecutor(1)
        self.gen = {}
        self.inflight = 0
        self.done.connect(lambda cb, r: cb(r))
        self.finished.connect(self._dec)

    def _dec(self):
        self.inflight -= 1

    def cancel(self, key):
        self.gen[key] = self.gen.get(key, 0) + 1

    def submit(self, key, fn, cb):
        g = self.gen[key] = self.gen.get(key, 0) + 1
        stale = lambda: self.gen[key] != g
        self.inflight += 1

        def run():
            try:
                if not stale():
                    r = fn(stale)
                    if not stale():
                        self.done.emit(cb, r)
            except Exception as e:
                traceback.print_exc()
                self.failed.emit(f"{type(e).__name__}: {e}")
            finally:
                self.finished.emit()

        self.pool.submit(run)


def load_image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode == "I" or im.mode.startswith("I;16"):  # convert("RGB") would clip 16/32-bit to white
            a = np.asarray(im)
            a = (a >> 8).astype(np.uint8) if im.mode.startswith("I;16") else np.clip(a // 257, 0, 255).astype(np.uint8)
            return np.dstack([a] * 3)
        return np.asarray(im.convert("RGB")).copy()


def qimage_to_rgb(img: QImage):
    if img.isNull():
        return None
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    a = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes()).reshape(h, bpl)
    return a[:, :w * 3].reshape(h, w, 3).copy()


def encode_png(arr):
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError("PNG encoding failed")
    return buf.tobytes()


def write_png(path, arr):
    Path(path).write_bytes(encode_png(arr))  # imencode + write handles non-ASCII paths on Windows


def labeled_slider(text, lo, hi, val, on_change, fmt="{}"):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(4, 0, 4, 0)
    s = QSlider(Qt.Horizontal, minimum=lo, maximum=hi, value=val)
    s.setFixedWidth(110)
    v = QLabel(fmt.format(val))
    v.setMinimumWidth(32)
    s.valueChanged.connect(lambda x: (v.setText(fmt.format(x)), on_change()))
    lay.addWidget(QLabel(text))
    lay.addWidget(s)
    lay.addWidget(v)
    return w, s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CutOut")
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        self.engine = Engine()
        self.models, self.cpu = Runner(), Runner()
        self.rgb = self.rgb_proxy = self.src_path = None
        self.image_id = 0
        self.history, self.redo_stack = [], []
        self.candidate = None
        self.points, self.labels, self.box = [], [], None
        self.key_pick = None
        self.paint = None  # uint8 paint mask at proxy resolution
        self._last_paint = None
        self._dist = (None, None)  # (pick, delta-E map), written by the cpu worker
        self._subj = None  # last Auto subject detection, for Reach
        self._asking = False  # a model-download question is open
        self._latest = None  # (alpha, rgba) at full res, written by the cpu worker
        self.last_dir = Path.home()
        self._build_ui()
        for r in (self.models, self.cpu):
            r.failed.connect(self._failed)
            r.status.connect(lambda m: self.statusBar().showMessage(m))
            r.finished.connect(self._update_status)
        sb = self.statusBar()
        sb.messageChanged.connect(lambda m: m.startswith("Error: ") or sb.setToolTip(""))
        self._update_status()
        if not is_cached("BiRefNet"):
            sb.showMessage("Models are not downloaded yet (~1.5 GB total). You will be asked before each one is fetched.")
        else:  # load while the user picks an image: imports alone take ~8 s on Windows
            self.models.submit("prep", lambda s: self.engine.use("BiRefNet", self.models.status.emit), lambda r: None)
        self._update_actions()

    def _failed(self, m):
        if m.startswith("ModelMissing: "):
            name = m.removeprefix("ModelMissing: ")
            if self._asking:  # one question at a time; the Yes callback retries the dropped action
                return
            self._asking = True
            ans = QMessageBox.question(
                self, "Model not downloaded",
                f"{name} (~{DOWNLOAD_MB[name]} MB) is not downloaded yet.\n"
                "Download it now into models/? Needs internet; first time only.",
                QMessageBox.Yes | QMessageBox.No)
            self._asking = False
            if ans == QMessageBox.Yes:  # then redo what failed: auto-detect / prep (Refine is pressed again)
                self.models.submit("download", lambda s: self.engine.download(name, self.models.status.emit),
                                   self._guard(lambda r: (self.mode_changed(), self.refresh())))
            else:
                self.statusBar().showMessage(f"{name} not downloaded — the action was skipped", 6000)
            return
        self.statusBar().showMessage("Error: " + m.partition("\n")[0])  # stays until the next message
        self.statusBar().setToolTip(m)
        if "Could not load" in m:  # engine.use() tags model load/download failures
            QMessageBox.warning(self, "Model could not be loaded", m)

    def closeEvent(self, e):
        for r in (self.models, self.cpu):  # don't wait for a running download or render
            r.pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(e)

    # ================= UI =================

    def _build_ui(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addAction(self._act("Open…", self.open_dialog, QKeySequence.Open))
        tb.addAction(self._act("Paste", self.paste, QKeySequence.Paste))
        tb.addSeparator()
        tb.addWidget(QLabel(" Mode "))
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        tb.addWidget(self.mode)
        self.refine = self._act("Refine edges", self.run_refine)
        self.refine.setToolTip("Re-matte the applied selection with BiRefNet-matting (hair, branches)")
        tb.addAction(self.refine)
        tb.addSeparator()
        exp = self.export_btn = QToolButton()
        exp.setText("Export")
        exp.setPopupMode(QToolButton.MenuButtonPopup)
        menu = self.export_menu = QMenu(exp)
        menu.addAction(self._act("Export PNG…", self.export, "Ctrl+E"))
        self.crop = menu.addAction("Crop to mask bounds")
        self.crop.setCheckable(True)
        menu.addAction(self._act("Save mask as PNG…", lambda: self.export(mask_only=True)))
        menu.addAction(self._act("Copy result to clipboard", self.copy_result, "Ctrl+Shift+C"))
        exp.setToolTip(menu.actions()[0].toolTip())
        exp.setMenu(menu)
        exp.clicked.connect(self.export)
        tb.addWidget(exp)

        self.addToolBarBreak()
        sel = self.addToolBar("Selection")
        sel.setMovable(False)
        self.panels = QStackedWidget()
        self.detect_btn = self._button("Detect subject", lambda: self.run_subject())
        reach, self.reach = labeled_slider("Reach", -10, 10, 0, self.run_reach, "{:+}")
        reach.setToolTip("Right takes in parts the model was unsure about, left keeps only what it is sure of")
        self.panels.addWidget(self._panel(
            QLabel("Drag a rectangle to limit detection · Esc clears it "),
            reach, self.detect_btn))
        brush, self.brush = labeled_slider("Brush", 5, 200, 40, self._brush_changed, "{} px")
        self.panels.addWidget(self._panel(
            QLabel("Paint over the area to extract · Right-drag or Alt-drag erases "),
            brush, self._button("Clear paint", self.reset_session)))
        self.panels.addWidget(self._panel(
            QLabel("Left click: include · Right click: exclude · Drag: box "),
            self._button("Reset prompts", self.reset_session)))
        tol, self.tol = labeled_slider("Tolerance", 0, 100, 12, lambda: self.run_key())
        soft, self.soft = labeled_slider("Softness", 1, 60, 10, lambda: self.run_key())
        self.invert = QCheckBox("Invert")
        self.invert.toggled.connect(lambda: self.run_key())
        self.panels.addWidget(self._panel(QLabel("Click a colour to key out · Esc clears it "), tol, soft, self.invert))
        sel.addWidget(self.panels)
        sel.addSeparator()
        sel.addWidget(QLabel(" Combine "))
        self.op = QComboBox()
        self.op.addItems(OPS)
        self.op.setToolTip("How the next selection is merged with the current mask")
        self.op.currentIndexChanged.connect(self._restate)
        sel.addWidget(self.op)
        self.apply_act = self._act("Apply", self.apply, ["Return", "Enter"])
        # on macOS StandardKey is Cmd+Z; also accept the physical Control key (Meta there)
        mac = ["Meta+Z"] if sys.platform == "darwin" else []
        self.undo_act = self._act("Undo", self.undo, [*QKeySequence.keyBindings(QKeySequence.Undo), *mac])
        mac = ["Meta+Shift+Z"] if sys.platform == "darwin" else []
        self.redo_act = self._act("Redo", self.redo, [*QKeySequence.keyBindings(QKeySequence.Redo), *mac])
        for a in (self.apply_act, self.undo_act, self.redo_act):
            sel.addAction(a)
        self.addAction(self._act("Reset", self.reset_session, Qt.Key_Escape))
        for i, m in enumerate(MODES):
            self.addAction(self._act(m, lambda i=i: self.mode.setCurrentIndex(i), str(i + 1)))
        self.addAction(self._act("Smaller brush", lambda: self.brush.setValue(self.brush.value() - 5), "["))
        self.addAction(self._act("Larger brush", lambda: self.brush.setValue(self.brush.value() + 5), "]"))

        self.addToolBarBreak()
        fin = self.addToolBar("Finish")
        fin.setMovable(False)
        w, self.feather = labeled_slider("Feather", 0, 50, 0, self.refresh, "{} px")
        fin.addWidget(w)
        w, self.grow = labeled_slider("Shrink/Grow", -50, 50, 0, self.refresh, "{} px")
        fin.addWidget(w)
        w, self.defringe = labeled_slider("Defringe", 0, 100, 0, self.refresh, "{} %")
        fin.addWidget(w)

        self.source, self.preview = ImageView(), ImageView()
        self.source.set_background(QColor(40, 40, 40))
        key = lambda k: QKeySequence(k).toString(QKeySequence.NativeText)
        self.source.hint = f"Drop an image here · {key(QKeySequence.Open)} to open · {key(QKeySequence.Paste)} to paste"
        self.preview.set_tool("view")  # the preview only zooms and pans
        self.addAction(self._act("Fit source", lambda: self.source.fit(), "Ctrl+0"))
        self.addAction(self._act("Source at 100%", lambda: self.source.set_zoom(1), "Ctrl+1"))
        self.source.clicked.connect(self.on_click)
        self.source.boxDrawn.connect(self.on_box)
        self.source.painted.connect(self.on_paint)
        self.source.strokeDone.connect(self.run_paint)
        self.link = QCheckBox("Link views", checked=True)
        self.link.toggled.connect(lambda on: on and self.preview.sync_from(self.source))
        self.source.viewChanged.connect(lambda: self._sync(self.source, self.preview))
        self.preview.viewChanged.connect(lambda: self._sync(self.preview, self.source))
        bg = QComboBox()
        bg.addItems(["Checkerboard", "White", "Black", "Custom…"])
        self._bg_index = 0
        bg.currentIndexChanged.connect(lambda i: self.set_bg(i, bg))

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._pane("Source", self.source,
                                   self._button("Fit", lambda: self.source.fit()),
                                   self._button("100%", lambda: self.source.set_zoom(1))))
        split.addWidget(self._pane("Preview", self.preview, bg, self.link,
                                   self._button("Fit", lambda: self.preview.fit()),
                                   self._button("100%", lambda: self.preview.set_zoom(1))))
        split.setSizes([10000, 10000])  # [1, 1] clamps to each pane's minimum width, so the wider Preview header won
        for i in (0, 1):  # keep the panes equal when the window is resized or maximized
            split.setStretchFactor(i, 1)
        self.setCentralWidget(split)

        sb = self.statusBar()
        self.busy = QProgressBar(maximum=0, maximumWidth=80, maximumHeight=12, textVisible=False)
        self.zoom_label, self.model_label = QLabel(), QLabel()
        sb.addPermanentWidget(self.busy)
        sb.addPermanentWidget(self.zoom_label)
        sb.addPermanentWidget(QLabel(f"Device: {device_label()}"))
        sb.addPermanentWidget(self.model_label)
        sb.showMessage("Open, paste or drop an image")

    def _act(self, text, slot, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(lambda *_: slot())
        if shortcut is not None:
            a.setShortcuts(shortcut) if isinstance(shortcut, list) else a.setShortcut(shortcut)
            a.setToolTip(f"{text.rstrip('…')} ({a.shortcut().toString(QKeySequence.NativeText)})")
        return a

    def _button(self, text, slot):
        b = QPushButton(text)
        b.clicked.connect(lambda *_: slot())
        return b

    def _panel(self, *widgets):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 0, 4, 0)
        for x in widgets:
            lay.addWidget(x)
        lay.addStretch()
        return w

    def _pane(self, title, view, *extras):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(6, 2, 6, 0)
        t = QLabel(f"<b>{title}</b>")
        head.addWidget(t)
        head.addStretch()
        for x in extras:
            head.addWidget(x)
        v.addLayout(head)
        v.addWidget(view)
        return w

    def _sync(self, src, dst):
        if self.link.isChecked():
            dst.sync_from(src)
        self.zoom_label.setText(f"Zoom {src.zoom() * 100:.0f}%")

    def set_bg(self, i, combo):
        if i == 3:
            c = QColorDialog.getColor(QColor(0, 160, 80), self, "Preview background")
            if not c.isValid():
                combo.blockSignals(True)
                combo.setCurrentIndex(self._bg_index)
                combo.blockSignals(False)
                return
            self.preview.set_background(c)
        else:
            self.preview.set_background(["checker", QColor("white"), QColor("black")][i])
        self._bg_index = i

    def _update_status(self):
        busy = self.models.inflight + self.cpu.inflight
        self.busy.setVisible(busy > 0)
        self.model_label.setText(f"Model: {self.engine.name or 'none'}")
        if not busy and self.statusBar().currentMessage().endswith("…"):
            self.statusBar().clearMessage()

    # ================= image loading =================

    def open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open image", str(self.last_dir), IMAGE_FILTER)
        if p:
            self.load_path(p)

    def load_path(self, p):
        self.statusBar().showMessage(f"Loading {Path(p).name}…")
        self.cpu.submit("load", lambda s: load_image(p), lambda rgb: self.set_image(rgb, Path(p)))

    def paste(self):
        md = QGuiApplication.clipboard().mimeData()
        if md.hasImage():
            rgb = qimage_to_rgb(QGuiApplication.clipboard().image())
            if rgb is None:
                return self.statusBar().showMessage("Clipboard has no image", 4000)
            self.set_image(rgb, None)
        elif md.hasUrls() and md.urls()[0].isLocalFile():
            self.load_path(md.urls()[0].toLocalFile())
        else:
            self.statusBar().showMessage("Clipboard has no image", 4000)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and md.urls()[0].isLocalFile():
            self.load_path(md.urls()[0].toLocalFile())
        elif md.hasImage():
            rgb = qimage_to_rgb(QImage(md.imageData()))
            if rgb is None:
                return self.statusBar().showMessage("Clipboard has no image", 4000)
            self.set_image(rgb, None)

    def set_image(self, rgb, path):
        self.rgb = np.ascontiguousarray(rgb)
        self.rgb_proxy = make_proxy(self.rgb)
        self.src_path = path
        if path:
            self.last_dir = path.parent
        self.image_id += 1
        self.history, self.redo_stack = [], []
        self._latest = None
        self._dist = (None, None)
        self.reset_session(refresh=False)
        self.source.set_image(self.rgb, self.rgb_proxy, fit=True)
        self.preview.set_image(self.rgb, self.rgb_proxy, fit=True)
        h, w = rgb.shape[:2]
        self.setWindowTitle(f"CutOut — {path.name if path else 'pasted image'} ({w}×{h})")
        self.statusBar().showMessage(f"Loaded {w}×{h}", 3000)
        self.mode_changed()

    def _guard(self, cb):
        """Drop results that belong to a previous image."""
        iid = self.image_id
        return lambda r: cb(r) if iid == self.image_id else None

    # ================= modes =================

    def mode_name(self):
        return self.mode.currentText()

    def mode_changed(self):
        if self.candidate is not None:
            self.apply()
        self.reset_session(refresh=False)
        mode = self.mode_name()
        self.panels.setCurrentIndex(self.mode.currentIndex())
        self.source.set_tool({"Paint region": "paint", "Color key": "none"}.get(mode, "box"))
        if self.rgb is None:
            return
        rgb, iid, st = self.rgb, self.image_id, self.models.status.emit
        if mode == "Auto subject" and not self.history:
            self.run_subject()
        elif mode in ("Auto subject", "Paint region"):
            self.models.submit("prep", lambda s: self.engine.use("BiRefNet", st), lambda r: None)
        elif mode == "Click select":
            self.models.submit("prep", lambda s: self.engine.sam_prepare(iid, rgb, st), lambda r: None)
        else:  # colour key needs no model: free the memory
            self.models.submit("prep", lambda s: self.engine.unload(), lambda r: None)
        self.refresh()

    def on_click(self, x, y, button):
        if self.rgb is None:
            return
        H, W = self.rgb.shape[:2]
        x, y = min(x, W - 0.5), min(y, H - 0.5)  # the view's scene rect includes the far edge
        mode = self.mode_name()
        if mode == "Click select":
            self.points.append((x, y))
            self.labels.append(1 if button == Qt.LeftButton else 0)
            self.run_sam()
        elif mode == "Color key" and button == Qt.LeftButton:
            self.key_pick = (int(x), int(y))
            self.run_key()

    def on_box(self, r: QRectF):
        box = (int(r.left()), int(r.top()), int(np.ceil(r.right())), int(np.ceil(r.bottom())))
        mode = self.mode_name()
        if mode == "Auto subject":
            self.run_subject(box)
        elif mode == "Click select":
            self.box = box
            self.run_sam()

    def _brush_changed(self):
        self.source.brush = self.brush.value()

    def on_paint(self, x, y, erase):
        """Paint into a proxy-resolution mask; the region is coarse by nature, the model finds the edges."""
        if self.paint is None:
            self.paint = np.zeros(self.rgb_proxy.shape[:2], np.uint8)
        s = self.rgb_proxy.shape[1] / self.rgb.shape[1]
        p = (round(x * s), round(y * s))
        r = max(1, round(self.brush.value() / self.source.zoom() * s))
        last = self._last_paint if self._last_paint and self._last_paint[2] == erase else (*p, erase)
        cv2.line(self.paint, last[:2], p, 0 if erase else 255, 2 * r, cv2.LINE_AA)
        cv2.circle(self.paint, p, r, 0 if erase else 255, -1, cv2.LINE_AA)
        self._last_paint = (*p, erase)
        self.source.set_paint(self.paint)

    def run_paint(self):
        self._last_paint = None
        self._update_actions()
        if self.paint is None or not self.paint.any():
            self.models.cancel("paint")
            self.candidate = None
            return self.refresh()
        rgb, H, W = self.rgb, *self.rgb.shape[:2]
        paint = self.paint.copy()

        def job(stale):
            region = cv2.resize(paint, (W, H), interpolation=cv2.INTER_LINEAR)
            return self.engine.subject(rgb, region=region, on_status=self.models.status.emit)

        self.models.submit("paint", job, self._guard(self._set_candidate))

    def _overlay(self):
        pts = [(x, y, "#2ecc40" if l else "#ff4136") for (x, y), l in zip(self.points, self.labels)]
        if self.key_pick:
            x, y = self.key_pick
            r, g, b = self.rgb[y, x]
            pts.append((x + 0.5, y + 0.5, QColor(int(r), int(g), int(b))))
        self.source.set_overlay(pts, self.box)
        self._update_actions()

    def run_subject(self, box=None):
        if self.rgb is None:
            return
        rgb = self.rgb
        if box is None:  # whole image: commits directly
            self.reset_session(refresh=False)
        else:  # limited to a box: a candidate like every other prompt
            self.box = box
            self._overlay()
        base, op, reach = self.base(), self.op_name(), self.reach.value() / 2

        def job(stale):
            raw = self.engine.subject_raw(rgb, box, on_status=self.models.status.emit)
            return raw, subject_mask(rgb, raw, reach)

        def done(r):
            if box is None:
                if not self._subject_done(base, op, r[1]):
                    return
            else:
                self._set_candidate(r[1])
            out = self.candidate if box is not None else self.base()
            self._subj = dict(raw=r[0], rgb=rgb, base=base, op=op, out=out)  # kept for Reach changes

        self.models.submit("subject" if box is None else "box", job, self._guard(done))

    def _subject_done(self, base, op, m):
        if self.base() is not base:  # undo/redo happened while detecting: the result no longer fits
            return False
        self.commit(maskops.combine(base, m, op))
        if not (m > 127).any():
            self.statusBar().showMessage("No clear subject found: try Paint region or drag a rectangle", 8000)
        return True

    def run_reach(self):
        """Redo the last detection's mask at the new Reach, in place, while it is still the current result."""
        s = self._subj
        if s is None or not (self.candidate is s["out"] or self.base() is s["out"]):
            return
        reach, pending = self.reach.value() / 2, self.candidate is s["out"]

        def job(stale):
            m = subject_mask(s["rgb"], s["raw"], reach)
            return m if pending else maskops.combine(s["base"], m, s["op"])

        def done(m):
            if pending and self.candidate is s["out"]:
                s["out"] = m
                self._set_candidate(m)
            elif not pending and self.base() is s["out"]:
                s["out"] = self.history[-1] = m  # same undo step as the detection
                self.redo_stack = []
                self.refresh()

        self.cpu.submit("reach", job, self._guard(done))

    def run_sam(self):
        self._overlay()
        if not self.points and self.box is None:
            self.models.cancel("sam")
            self.candidate = None
            return self.refresh()
        rgb, iid, pts, lbl, box = self.rgb, self.image_id, list(self.points), list(self.labels), self.box
        self.models.submit("sam", lambda s: self.engine.sam(iid, rgb, pts, lbl, box, self.models.status.emit),
                           self._guard(self._set_candidate))

    def run_key(self):
        if self.rgb is None or self.key_pick is None:
            return
        self._overlay()
        rgb, pick, iid = self.rgb, self.key_pick, self.image_id
        tol, soft, inv = self.tol.value(), self.soft.value(), self.invert.isChecked()

        def job(stale):
            if self._dist[0] != (iid, pick):
                self._dist = ((iid, pick), maskops.color_distance(rgb, *pick))
            return maskops.key_alpha(self._dist[1], tol, soft, inv)

        self.cpu.submit("key", job, self._guard(self._set_candidate))

    def _set_candidate(self, m):
        self.candidate = m
        msg = PENDING
        if self.history and self.op_name() == "Replace":
            msg += " (Combine: Replace — pick Add/Subtract/Intersect to keep the current mask)"
        if self.mode_name() == "Auto subject" and not (m > 127).any():
            self.statusBar().showMessage("No clear subject found: try Paint region or drag a rectangle", 8000)
        else:
            self.statusBar().showMessage(msg)
        self.refresh()

    def _restate(self):
        """Re-render; with a candidate pending, also update its status line."""
        self._set_candidate(self.candidate) if self.candidate is not None else self.refresh()

    # ================= mask history =================

    def base(self):
        return self.history[-1] if self.history else None

    def op_name(self):
        return self.op.currentText()

    def commit(self, mask):
        limit = min(HISTORY, max(3, 400_000_000 // mask.nbytes))  # 30 masks, fewer on huge images
        self.history = (self.history + [mask])[-limit:]
        self.redo_stack = []
        self.refresh()

    def apply(self):
        if self.candidate is not None:
            self.commit(maskops.combine(self.base(), self.candidate, self.op_name()))
        self.reset_session()

    def run_refine(self):
        """Re-matte the committed mask; the result is a new history step, so Undo reverts it."""
        if self.candidate is not None:  # a pending candidate is committed first, as on a mode switch
            self.apply()
        base, rgb = self.base(), self.rgb
        if base is None:
            return self.statusBar().showMessage("Nothing to refine yet: make a selection first", 5000)
        self.models.submit("refine", lambda s: self.engine.refine(rgb, base, self.models.status.emit),
                           self._guard(lambda r: self.base() is base and self.commit(r)))

    def reset_session(self, refresh=True):
        self.points, self.labels, self.box, self.key_pick = [], [], None, None
        self.paint = self._last_paint = None
        self.source.set_paint(None)
        self.candidate = None
        self.models.cancel("sam")  # drop in-flight prompt results
        self.models.cancel("paint")
        self.models.cancel("box")
        self.cpu.cancel("key")
        self.source.set_overlay()
        if refresh:
            self.refresh()

    def undo(self):
        """Step back one prompt (last click, then box), drop a colour pick or paint, then step back masks."""
        if self.points or self.box is not None:
            if self.points:
                self.points.pop()
                self.labels.pop()
            else:
                self.box = None
                self.models.cancel("box")
            return self.run_sam()  # with no prompts left this just clears the candidate
        if self.candidate is not None or self.key_pick or self.paint is not None:
            return self.reset_session()
        if self.history:
            self.redo_stack.append(self.history.pop())
            self.refresh()

    def redo(self):
        if self.redo_stack:
            self.history.append(self.redo_stack.pop())
            self.refresh()

    # ================= rendering =================

    def _update_actions(self):
        img = self.rgb is not None
        prompts = self.points or self.box is not None or self.key_pick or self.paint is not None
        self.apply_act.setEnabled(self.candidate is not None)
        self.undo_act.setEnabled(bool(self.history or prompts or self.candidate is not None))
        self.redo_act.setEnabled(bool(self.redo_stack))
        self.refine.setEnabled(bool(self.history or self.candidate is not None))
        for w in (self.export_btn, *self.export_menu.actions(), self.detect_btn):
            w.setEnabled(img)
        if self.candidate is None and self.statusBar().currentMessage().startswith(PENDING):
            self.statusBar().clearMessage()

    def refresh(self):
        self._update_actions()
        if self.rgb is None:
            return
        base, cand, op = self.base(), self.candidate, self.op_name()
        rgb, rgbp = self.rgb, self.rgb_proxy
        params = self.grow.value(), self.feather.value(), self.defringe.value() / 100

        def job(stale):
            m = maskops.combine(base, cand, op) if cand is not None else base
            if m is None:
                self._latest = None
                return None
            s = rgbp.shape[1] / rgb.shape[1]
            mp = m if s == 1 else cv2.resize(m, (rgbp.shape[1], rgbp.shape[0]), interpolation=cv2.INTER_AREA)
            prgb, pa = maskops.finish(rgbp, mp, *params, scale=s)
            proxy = np.dstack([prgb, pa])
            if s == 1:
                self._latest = (pa, proxy)
                return proxy, proxy
            self.cpu.done.emit(self._guard(self._show_proxy), proxy)  # fast feedback before full res
            if stale():
                return None
            frgb, fa = maskops.finish(rgb, m, *params)
            full = np.dstack([frgb, fa])
            self._latest = (fa, full)
            return full, proxy

        self.cpu.submit("render", job, self._guard(self._rendered))

    def _show_proxy(self, proxy):
        if self.preview.full is not None and self.preview.full.shape[:2] == self.rgb.shape[:2]:
            self.preview.proxy = proxy
            self.preview._show(False)  # stays on proxy until the full-res result lands

    def _rendered(self, r):
        if r is None:
            self.preview.set_image(self.rgb, self.rgb_proxy)
        else:
            self.preview.set_image(*r)

    # ================= export =================

    def _final(self, crop):
        """(alpha, rgba) for the current state, optionally cropped. Runs on the cpu worker after any render."""
        if self._latest is None:
            raise ValueError("No selection yet")
        a, rgba = self._latest
        if crop:
            b = maskops.bbox(a, 2)
            if b:
                a, rgba = np.ascontiguousarray(a[b[1]:b[3], b[0]:b[2]]), np.ascontiguousarray(rgba[b[1]:b[3], b[0]:b[2]])
        return a, rgba

    def _has_result(self):
        if self._latest is None or not self._latest[0].any():
            self.statusBar().showMessage("Nothing to export yet", 4000)
            return False
        return True

    def export(self, mask_only=False):
        if not self._has_result():
            return
        stem = self.src_path.stem if self.src_path else "pasted"
        default = self.last_dir / f"{stem}_{'mask' if mask_only else 'cutout'}.png"
        p, _ = QFileDialog.getSaveFileName(self, "Save mask" if mask_only else "Export PNG", str(default), "PNG (*.png)")
        if not p:
            return
        p = Path(p)
        if p.suffix.lower() != ".png":
            p = p.with_name(p.name + ".png")
        if self.src_path and p.resolve() == self.src_path.resolve():
            self.statusBar().showMessage("Refusing to overwrite the source image — pick another name", 6000)
            return

        crop = self.crop.isChecked()

        def job(stale):
            a, rgba = self._final(crop)
            write_png(p, a if mask_only else rgba)
            return p

        self.statusBar().showMessage(f"Saving {p.name}…")
        self.cpu.submit("export", job, lambda p: self.statusBar().showMessage(f"Saved {p}", 6000))

    def copy_result(self):
        def job(stale):
            rgba = self._final(crop)[1]
            return rgba, encode_png(rgba)

        def done(r):
            md = QMimeData()
            md.setImageData(to_qimage(r[0]).copy())
            md.setData("image/png", r[1])
            # Windows apps read alpha only from the registered "PNG" format; the bitmap formats lose it
            md.setData('application/x-qt-windows-mime;value="PNG"', r[1])
            QGuiApplication.clipboard().setMimeData(md)
            self.statusBar().showMessage("Result copied to clipboard", 4000)
        if not self._has_result():
            return
        crop = self.crop.isChecked()
        self.cpu.submit("copy", job, done)
