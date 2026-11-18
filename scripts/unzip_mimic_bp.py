from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = Path(args.data_dir)
    zips = sorted(root.glob("*.zip"))
    if not zips:
        print(f"No zip files found in {root}")
        return

    for z in zips:
        out = root / z.stem
        if out.exists() and not args.overwrite:
            print(f"Skipping {z.name}, folder already exists: {out}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {z.name} -> {out}")
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(out)
    print("Done.")


if __name__ == "__main__":
    main()
