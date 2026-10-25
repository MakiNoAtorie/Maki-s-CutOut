import os
import sys

WIN_DATA = os.path.join(os.environ.get("LOCALAPPDATA", ""), "CutOut")  # Windows shortcut: icon and log
if sys.stdout is None:  # pythonw has no console; tqdm and tracebacks need a stream
    os.makedirs(WIN_DATA, exist_ok=True)
    sys.stdout = sys.stderr = open(os.path.join(WIN_DATA, "CutOut.log"), "a", buffering=1, encoding="utf-8")

import engine.device  # noqa: E402,F401  sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads
from PySide6.QtGui import QIcon  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CutOut")
    if sys.platform == "win32":
        app.setWindowIcon(QIcon(os.path.join(WIN_DATA, "CutOut.ico")))  # made by make_windows_shortcut.ps1
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load_path(sys.argv[1])
    os._exit(app.exec())  # a running model job must not keep the process alive


if __name__ == "__main__":
    main()
