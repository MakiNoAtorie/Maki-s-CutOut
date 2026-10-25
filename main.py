import os
import sys

if sys.stdout is None:  # pythonw (Windows shortcut) has no console; tqdm and tracebacks need a stream
    LOG = os.path.join(os.environ.get("LOCALAPPDATA", ""), "CutOut")
    os.makedirs(LOG, exist_ok=True)
    sys.stdout = sys.stderr = open(os.path.join(LOG, "CutOut.log"), "a", buffering=1, encoding="utf-8")

import engine.device  # noqa: E402,F401  sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CutOut")
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "icon.svg")))
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load_path(sys.argv[1])
    os._exit(app.exec())  # a running model job must not keep the process alive


if __name__ == "__main__":
    main()
