from pathlib import Path
import csv
from typing import Callable, Iterable

from PIL import Image


def initialize_csv_logger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["u", "v", "mask_value", "accepted", "triangle_index"])


def log_csv_sample(path: Path, u: float, v: float, mask_value: float, accepted: bool, triangle_index: int) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([f"{u:.8f}", f"{v:.8f}", f"{mask_value:.6f}", int(accepted), triangle_index])


def save_callable_mask_as_image(mask_fn: Callable[[float, float], float], path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        v = (y + 0.5) / size
        for x in range(size):
            u = (x + 0.5) / size
            val = mask_fn(u, v)
            val = 0.0 if val < 0.0 else 1.0 if val > 1.0 else val
            px[x, y] = int(val * 255.0 + 0.5)
    img.save(str(path))


def save_combined_mask_image(surface, path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (size, size))
    px = img.load()
    # Combined value = product/add/subtract per surface.apply_masks
    for y in range(size):
        v = (y + 0.5) / size
        for x in range(size):
            u = (x + 0.5) / size
            val = surface.apply_masks(u, v)
            val = 0.0 if val < 0.0 else 1.0 if val > 1.0 else val
            px[x, y] = int(val * 255.0 + 0.5)
    img.save(str(path))
