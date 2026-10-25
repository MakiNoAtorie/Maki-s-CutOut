"""Render app/icon.svg to OUT.ico (Windows) or OUT.iconset (macOS iconutil folder). Usage: python make_icon.py OUT"""
import os
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

SVG = str(Path(__file__).resolve().parent.parent / "app" / "icon.svg")


def render(size):
    img = QImage(size, size, QImage.Format_RGBA8888)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(SVG).render(p)
    p.end()
    return Image.frombytes("RGBA", (size, size), bytes(img.constBits()))


def main():
    out = sys.argv[1]
    QGuiApplication([sys.argv[0], "-platform", "offscreen"])
    if out.endswith(".ico"):
        sizes = (256, 128, 64, 48, 32, 24, 16)  # each size drawn from the vector, not downscaled
        imgs = [render(s) for s in sizes]
        imgs[0].save(out, sizes=[(s, s) for s in sizes], append_images=imgs[1:])
    else:
        os.makedirs(out)
        for s in (16, 32, 128, 256, 512):
            render(s).save(f"{out}/icon_{s}x{s}.png")
            render(s * 2).save(f"{out}/icon_{s}x{s}@2x.png")


main()
