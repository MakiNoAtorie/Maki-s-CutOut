import os
import sys

import engine.device  # noqa: F401  sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads
from PySide6.QtWidgets import QApplication

from app.window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CutOut")
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load_path(sys.argv[1])
    os._exit(app.exec())  # a running model job must not keep the process alive


if __name__ == "__main__":
    main()
