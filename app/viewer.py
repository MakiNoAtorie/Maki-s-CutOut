"""Zoom/pan image view with a downscaled proxy and lazy full-resolution pixmap."""
import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

PROXY = 2048


def to_qimage(arr):
    """Wrap a contiguous uint8 HxWx3/4 array. Caller keeps arr alive (or copies the QImage)."""
    h, w, c = arr.shape
    fmt = QImage.Format_RGBA8888 if c == 4 else QImage.Format_RGB888
    return QImage(arr.data, w, h, arr.strides[0], fmt)


_CHECKER = []


def _checker():
    if not _CHECKER:
        pm = QPixmap(24, 24)
        pm.fill(QColor(205, 205, 205))
        p = QPainter(pm)
        p.fillRect(0, 0, 12, 12, QColor(245, 245, 245))
        p.fillRect(12, 12, 12, 12, QColor(245, 245, 245))
        p.end()
        _CHECKER.append(pm)
    return _CHECKER[0]


def make_proxy(arr):
    h, w = arr.shape[:2]
    s = PROXY / max(h, w)
    if s >= 1:
        return arr
    return cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


class ImageView(QGraphicsView):
    viewChanged = Signal()
    clicked = Signal(float, float, object)  # image x, y, Qt.MouseButton
    boxDrawn = Signal(QRectF)  # image coords
    painted = Signal(float, float, bool)  # image x, y, erase — while dragging with the paint tool
    strokeDone = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.scene().addItem(self.item)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.background = "checker"
        self.hint = ""  # drawn while no image is loaded
        self.full = self.proxy = None
        self._full_pix = None
        self._showing_full = False
        self._space = False
        self._pan = None
        self._press = None
        self._band = None
        self.overlay = []
        self.tool = "box"  # left-drag: "box" draws a rectangle, "paint" paints, anything else: clicks only
        self.brush = 40  # paint radius in screen px
        self._painting = None
        self.paint_item = QGraphicsPixmapItem()
        self.paint_item.setZValue(1)
        self.scene().addItem(self.paint_item)
        self.cursor_item = QGraphicsEllipseItem()
        self.cursor_item.setPen(QPen(Qt.white, 1, Qt.DashLine))
        self.cursor_item.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations)
        self.cursor_item.setZValue(3)
        self.cursor_item.hide()
        self.scene().addItem(self.cursor_item)
        self._settle = QTimer(self, singleShot=True, interval=200, timeout=self._update_lod)
        self.horizontalScrollBar().valueChanged.connect(self.viewChanged)
        self.verticalScrollBar().valueChanged.connect(self.viewChanged)

    # ---------- image ----------

    def set_image(self, full, proxy=None, fit=False):
        self.full = full
        self.proxy = make_proxy(full) if proxy is None else proxy
        self._full_pix = None
        h, w = full.shape[:2]
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        self._show(False)
        if fit:
            self.fit()
        self._update_lod()

    def set_tool(self, tool):
        self.tool = tool
        self.cursor_item.setVisible(False)
        self._restore_cursor()

    def _restore_cursor(self):
        if self._space:
            c = Qt.OpenHandCursor
        elif self.tool == "paint" and self.full is not None:
            c = Qt.BlankCursor  # the brush circle is the cursor
        elif self.tool in ("box", "none"):
            c = Qt.CrossCursor
        else:
            return self.viewport().unsetCursor()
        self.viewport().setCursor(c)

    def set_paint(self, mask):
        """Show a paint mask (any resolution, stretched over the image) as a tinted overlay; None hides it."""
        if mask is None or self.full is None:
            self.paint_item.setPixmap(QPixmap())
            return
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 60, 200
        rgba[..., 3] = mask // 2
        self.paint_item.setPixmap(QPixmap.fromImage(to_qimage(rgba)))
        self.paint_item.setScale(self.full.shape[1] / mask.shape[1])

    def clear(self):
        self.full = self.proxy = self._full_pix = None
        self.item.setPixmap(QPixmap())

    def _show(self, full):
        if full:
            if self._full_pix is None:
                self._full_pix = QPixmap.fromImage(to_qimage(self.full))
            self.item.setPixmap(self._full_pix)
            self.item.setScale(1.0)
        else:
            self.item.setPixmap(QPixmap.fromImage(to_qimage(self.proxy)))
            self.item.setScale(self.full.shape[1] / self.proxy.shape[1])
        self._showing_full = full

    def zoom(self):
        return self.transform().m11()

    def _update_lod(self):
        if self.full is None:
            return
        proxy_scale = self.proxy.shape[1] / self.full.shape[1]
        want_full = proxy_scale < 1 and self.zoom() > proxy_scale * 1.01 and not self._settle.isActive()
        if want_full != self._showing_full:
            self._show(want_full)
        # nearest-neighbour when magnified so individual pixels stay visible
        self.item.setTransformationMode(Qt.FastTransformation if self.zoom() >= 2 else Qt.SmoothTransformation)

    def fit(self):
        if self.full is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self._update_lod()
            self.viewChanged.emit()

    def set_zoom(self, z):
        self.setTransform(self.transform().fromScale(z, z))
        self._update_lod()
        self.viewChanged.emit()

    def sync_from(self, other):
        self.blockSignals(True)
        self.setTransform(other.transform())
        self.centerOn(other.mapToScene(other.viewport().rect().center()))
        self.blockSignals(False)
        self._update_lod()

    def _zoom_by(self, f):
        if 0.01 < self.zoom() * f < 64:
            self.scale(f, f)
        self._settle.start()  # proxy while zooming, full res once it settles
        self._update_lod()
        self.viewChanged.emit()

    def wheelEvent(self, e):
        # trackpad scroll pans; mouse wheel and Ctrl/Cmd+scroll zoom
        if e.source() != Qt.MouseEventNotSynthesized and not e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            d = e.pixelDelta() if not e.pixelDelta().isNull() else e.angleDelta()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - d.y())
            return
        self._zoom_by(1.0015 ** e.angleDelta().y())

    def viewportEvent(self, e):
        if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture:  # trackpad pinch
            self._zoom_by(1 + e.value())
            return True
        return super().viewportEvent(e)

    # ---------- background ----------

    def set_background(self, bg):
        self.background = bg
        self.viewport().update()

    def drawBackground(self, p, rect):
        p.save()
        p.resetTransform()
        r = self.viewport().rect()
        if isinstance(self.background, QColor):
            p.fillRect(r, self.background)
        elif self.background == "checker":
            p.fillRect(r, QBrush(_checker()))
        if self.full is None and self.hint:
            p.setPen(QColor(170, 170, 170))
            p.drawText(r, Qt.AlignCenter, self.hint)
        p.restore()

    # ---------- overlay (prompts) ----------

    def set_overlay(self, points=(), box=None):
        """points: [(x, y, color)], box: (x0, y0, x1, y1) in image px."""
        for it in self.overlay:
            self.scene().removeItem(it)
        self.overlay = []
        for x, y, color in points:
            dot = QGraphicsEllipseItem(-5, -5, 10, 10)
            dot.setBrush(QBrush(QColor(color)))
            dot.setPen(QPen(Qt.white, 1.5))
            dot.setPos(x, y)
            dot.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations)
            self.scene().addItem(dot)
            self.overlay.append(dot)
        if box is not None:
            r = QGraphicsRectItem(QRectF(QPointF(box[0], box[1]), QPointF(box[2], box[3])))
            pen = QPen(QColor("#ffcc00"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            r.setPen(pen)
            self.scene().addItem(r)
            self.overlay.append(r)

    # ---------- mouse / keys ----------

    def enterEvent(self, e):
        self.setFocus()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.cursor_item.hide()
        self._space = False  # a Space released outside the view never reaches keyReleaseEvent
        self._restore_cursor()
        super().leaveEvent(e)

    def focusOutEvent(self, e):
        self._space = False
        self._restore_cursor()
        super().focusOutEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._restore_cursor()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._restore_cursor()
            return
        super().keyReleaseEvent(e)

    def _brush_cursor(self, pos):
        on = self.tool == "paint" and self.full is not None and self._pan is None
        self.cursor_item.setVisible(on)
        if on:
            r = self.brush
            self.cursor_item.setRect(-r, -r, 2 * r, 2 * r)
            self.cursor_item.setPos(self.mapToScene(pos.toPoint()))

    def _paint_at(self, pos):
        p = self.mapToScene(pos.toPoint())
        self.painted.emit(p.x(), p.y(), self._painting)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton or (self._space and e.button() == Qt.LeftButton):
            self._pan = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if self.tool == "paint" and self.full is not None and e.button() in (Qt.LeftButton, Qt.RightButton):
            self._painting = e.button() == Qt.RightButton or bool(e.modifiers() & Qt.AltModifier)
            self._paint_at(e.position())
            return
        if self.full is not None and e.button() in (Qt.LeftButton, Qt.RightButton):
            self._press = (e.position(), e.button())

    def mouseMoveEvent(self, e):
        self._brush_cursor(e.position())
        if self._painting is not None:
            return self._paint_at(e.position())
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(d.y()))
            return
        if self._press and self._press[1] == Qt.LeftButton and self.tool == "box":
            start = self._press[0]
            if (e.position() - start).manhattanLength() > 5:
                rect = QRectF(self.mapToScene(start.toPoint()), self.mapToScene(e.position().toPoint())).normalized()
                if self._band is None:
                    self._band = QGraphicsRectItem()
                    pen = QPen(QColor("#ffcc00"), 1.5)
                    pen.setCosmetic(True)
                    self._band.setPen(pen)
                    self.scene().addItem(self._band)
                self._band.setRect(rect)

    def mouseReleaseEvent(self, e):
        if self._painting is not None:
            self._painting = None
            self.strokeDone.emit()
            return
        if self._pan is not None:
            self._pan = None
            self._restore_cursor()
            return
        if not self._press:
            return
        start, button = self._press
        self._press = None
        if self._band is not None:
            rect = self._band.rect().intersected(self.sceneRect())
            self.scene().removeItem(self._band)
            self._band = None
            if rect.width() >= 2 and rect.height() >= 2:
                self.boxDrawn.emit(rect)
            return
        if (e.position() - start).manhattanLength() > 5:  # a drag that drew no box is not a click
            return
        p = self.mapToScene(e.position().toPoint())
        if self.sceneRect().contains(p):
            self.clicked.emit(p.x(), p.y(), button)

    # Let drops reach the main window.
    def dragEnterEvent(self, e):
        e.ignore()

    def dragMoveEvent(self, e):
        e.ignore()

    def dropEvent(self, e):
        e.ignore()
