"""
00_build_firm_lists.py — Write firm ticker lists from catalog to data/firms/.

No WRDS dependency. The catalog contains curated ticker lists.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FIRMS_DIR
from src.catalog import get_all

FIRMS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    episodes = get_all()
    print(f"Writing firm lists for {len(episodes)} episodes...")

    for ep in episodes:
        out_path = FIRMS_DIR / f"{ep['id']}.csv"
        tickers = ep["tickers"]
        with open(out_path, "w") as f:
            f.write("ticker\n")
            for t in tickers:
                f.write(f"{t}\n")

    print(f"Done. {len(episodes)} files in {FIRMS_DIR}")


if __name__ == "__main__":
    main()
