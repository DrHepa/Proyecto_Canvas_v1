import sys
from pathlib import Path

def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller
        return Path(sys._MEIPASS)
    else:
        # Modo desarrollo
        return Path(__file__).resolve().parent
