import csv
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw


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
            val = max(0.0, min(1.0, val))
            px[x, y] = int(val * 255.0 + 0.5)
    img.save(str(path))


def save_combined_mask_image(surface, path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        v = (y + 0.5) / size
        for x in range(size):
            u = (x + 0.5) / size
            val = surface.apply_masks(u, v)
            val = max(0.0, min(1.0, val))
            px[x, y] = int(val * 255.0 + 0.5)
    img.save(str(path))


def save_uv_scatter_image(surface, path: Path, size: int = 1024, point_radius: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    grid_color = (45, 45, 45)
    mid_color = (70, 70, 70)
    
    for t in range(1, 10):
        x = int(size * (t / 10.0))
        y = x
        draw.line([(x, 0), (x, size)], fill=grid_color)
        draw.line([(0, y), (size, y)], fill=grid_color)
    
    draw.line([(size // 2, 0), (size // 2, size)], fill=mid_color)
    draw.line([(0, size // 2), (size, size // 2)], fill=mid_color)
    
    for p in surface.scatter_points:
        u, v = p.uv
        x = int(u * size + 0.5)
        y = int(v * size + 0.5)
        x = max(0, min(size - 1, x))
        y = max(0, min(size - 1, y))
        bbox = [x - point_radius, y - point_radius, x + point_radius, y + point_radius]
        draw.ellipse(bbox, fill=(200, 200, 255), outline=(255, 255, 255))
    
    img.save(str(path))