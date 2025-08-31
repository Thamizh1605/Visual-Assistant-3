# src/utils.py
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def ensure_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def model_path(name: str):
    ensure_models_dir()
    p = MODELS_DIR / name
    return p

def timestamped_filename(prefix="rec", ext="wav"):
    ts = int(time.time())
    return f"{prefix}_{ts}.{ext}"
