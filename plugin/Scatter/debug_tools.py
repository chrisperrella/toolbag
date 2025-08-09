import csv
import numpy as np
from PIL import Image


def save_callable_mask_as_image(mask_fn, filename, resolution=512):
    image = np.zeros((resolution, resolution), dtype=np.float32)
    for y in range(resolution):
        for x in range(resolution):
            u = x / resolution
            v = y / resolution
            value = float(mask_fn(u, v))
            image[y, x] = np.clip(value, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image, mode='L').save(filename)


def save_combined_mask_image(scatter_surface, filename, resolution=512):
    image = np.zeros((resolution, resolution), dtype=np.float32)
    for y in range(resolution):
        for x in range(resolution):
            u = x / resolution
            v = y / resolution
            value = float(scatter_surface.apply_masks(u, v))
            image[y, x] = np.clip(value, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image, mode='L').save(filename)


def initialize_csv_logger(csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['u', 'v', 'mask_value', 'accepted', 'triangle_index'])


def log_csv_sample(csv_path, u, v, mask_value, accepted, triangle_index):
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([u, v, mask_value, int(accepted), triangle_index])
