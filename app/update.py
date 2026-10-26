"""Find a newer release on GitHub and fetch its Windows installer. Self-check: python -m app.update"""
import hashlib
import json
import re
import tempfile
import urllib.request
from pathlib import Path

from . import VERSION

REPO = "MakiNoAtorie/Maki-s-CutOut"
ROOT = Path(__file__).resolve().parent.parent
TAG = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:-(\w+))?$")


def key(v):
    """Sort key: 0.5.0-Beta < 0.5.0-RC1 < 0.5.0 < 0.5.1-Beta."""
    a, b, c, pre = TAG.match(v).groups()
    return int(a), int(b), int(c), pre is None, (pre or "").lower()


def installed():
    return (ROOT / "unins000.exe").exists()  # written by Setup; absent in a git checkout


def latest():
    """Newest release newer than this one, pre-releases included: dict(tag, page, setup, sha256) or None."""
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases?per_page=30",
                                 headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        rels = [x for x in json.load(r) if not x["draft"] and TAG.match(x["tag_name"])]
    if not rels:
        return None
    rel = max(rels, key=lambda x: key(x["tag_name"]))
    if key(rel["tag_name"]) <= key(VERSION):
        return None
    exe = next((a for a in rel["assets"] if a["name"].lower().endswith("-setup.exe")), None)
    return {"tag": rel["tag_name"], "page": rel["html_url"],
            "setup": exe and exe["browser_download_url"],
            "sha256": exe and (exe.get("digest") or "").removeprefix("sha256:")}


def download(url, sha256):
    """Save the installer to the temp folder; refuse it if GitHub's published digest does not match."""
    path = Path(tempfile.gettempdir()) / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    if sha256 and hashlib.sha256(data).hexdigest() != sha256.lower():
        raise IOError("the downloaded installer does not match its checksum on GitHub")
    path.write_bytes(data)
    return path


if __name__ == "__main__":
    assert key("v0.5.0-Beta") < key("0.5.0-RC1") < key("0.5.0") < key("v0.5.1-beta") < key("1.0.0")
    assert key("v0.5.0-Beta") == key("0.5.0-beta")
    src = Path(tempfile.mkdtemp()) / "MakisCutOut-v9.9.9-Setup.exe"
    src.write_bytes(b"setup")
    good = hashlib.sha256(b"setup").hexdigest()
    assert download(src.as_uri(), good).read_bytes() == b"setup"
    try:
        download(src.as_uri(), "0" * 64)
        raise AssertionError("checksum mismatch was accepted")
    except IOError:
        pass
    print("ok; latest release newer than", VERSION, "->", latest())
