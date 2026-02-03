from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    image_dir = project_root / "data" / "Real-Images"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    tif_files = sorted(image_dir.glob("*.tif"))
    if not tif_files:
        print(f"No .tif files found in: {image_dir}")
        return

    size_counts: Counter[tuple[int, int]] = Counter()
    bad_files: list[Path] = []

    for tif_path in tif_files:
        try:
            with Image.open(tif_path) as img:
                w, h = img.size
                size_counts[(w, h)] += 1
                print(f"{tif_path.name}: {w}x{h}")
        except Exception:
            bad_files.append(tif_path)
            print(f"{tif_path.name}: Error")

    print("\nSummary")
    print(f"- Folder: {image_dir}")
    print(f"- TIF files: {len(tif_files)}")
    print(f"- Unique resolutions: {len(size_counts)}")

    for (w, h), n in sorted(size_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  - {w}x{h}: {n}")

    if bad_files:
        print(f"- Errors: {len(bad_files)} files could not be read")


if __name__ == "__main__":
	main()
