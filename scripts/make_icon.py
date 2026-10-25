"""App icon: rounded tile, checkerboard "transparency" with a solid subject cut out of it.
Usage: python make_icon.py OUT.ico | OUT.iconset (macOS iconutil folder)"""
import os
import sys

from PIL import Image, ImageDraw

out = sys.argv[1]
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
tile = Image.new("RGBA", (S, S), (235, 235, 240, 255))
d = ImageDraw.Draw(tile)
for y in range(0, S, 128):
    for x in range(0, S, 128):
        if (x + y) // 128 % 2:
            d.rectangle([x, y, x + 127, y + 127], fill=(200, 200, 210, 255))
d.ellipse([300, 180, 724, 604], fill=(40, 120, 230, 255))
d.rounded_rectangle([220, 560, 804, 900], 160, fill=(40, 120, 230, 255))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([100, 100, 924, 924], 185, fill=255)
img.paste(tile, (0, 0), mask)
if out.endswith(".ico"):
    img.save(out, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
else:
    os.makedirs(out)
    for s in (16, 32, 128, 256, 512):
        img.resize((s, s), Image.LANCZOS).save(f"{out}/icon_{s}x{s}.png")
        img.resize((s * 2, s * 2), Image.LANCZOS).save(f"{out}/icon_{s}x{s}@2x.png")
