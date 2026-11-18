from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--max_files", type=int, default=20)
    args = p.parse_args()

    root = Path(args.data_dir)
    print(f"Inspecting: {root.resolve()}")
    print(f"Exists: {root.exists()}")

    print("\nTop-level files/folders:")
    for item in sorted(root.iterdir()):
        if item.is_dir():
            n = sum(1 for _ in item.iterdir())
            print(f"  [DIR]  {item.name}  ({n} items)")
        else:
            print(f"  [FILE] {item.name}  ({item.stat().st_size/1024:.1f} KB)")

    print("\nPossible split files:")
    for name in ["train_subjects.txt", "val_subjects.txt", "validation_subjects.txt", "test_subjects.txt"]:
        matches = list(root.rglob(name))
        for m in matches:
            print(f"  {m}")

    print("\nExample files by folder:")
    for folder in sorted([x for x in root.iterdir() if x.is_dir()]):
        print(f"\n{folder.name}/")
        for i, f in enumerate(sorted(folder.iterdir())):
            if i >= args.max_files:
                print("  ...")
                break
            if f.is_file():
                print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
