"""Launch the Signal 9 Blind RF Signal Analysis GUI."""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Force pyqtgraph to bind to PySide6 (prevents QGuiApplication::font warnings)
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
os.environ["QT_API"] = "pyside6"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal9.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
