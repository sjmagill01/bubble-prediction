"""04_compute_measures.py — Compute firm-level vol measures for all episodes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vol_measures import compute_all

if __name__ == "__main__":
    compute_all()
