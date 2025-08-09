import bisect
import cProfile
import importlib.util
import math
import random
import sys
import time
import functools
import argparse
from pathlib import Path
from pstats import Stats
from typing import List, Tuple, Dict, Callable, Any
from contextlib import contextmanager
from debug_tools import (
    save_callable_mask_as_image,
    save_combined_mask_image,
    initialize_csv_logger,
    log_csv_sample,
)
import csv
import numpy as np
from PIL import Image

import mset

generate_primitives_path = Path("C:\\Program Files\\Marmoset\\Toolbag 5\\data\\plugin\\Generate Primitives")
sys.path.append(str(generate_primitives_path))
module_path = generate_primitives_path / "uvsphere.py"
spec = importlib.util.spec_from_file_location("uvsphere", module_path)
uvsphere_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uvsphere_module)
uvsphere = uvsphere_module.uvsphere


TIMING_MODE = 'log'

class TimingStats:
    def __init__(self):
        self.times: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
    
    def start(self, name: str) -> None:
        if TIMING_MODE == 'suppressed':
            return
        self.active_timers[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        if TIMING_MODE == 'suppressed':
            return 0.0
        
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was never started")
        
        duration = time.perf_counter() - self.active_timers[name]
        
        if name not in self.times:
            self.times[name] = []
        
        self.times[name].append(duration)
        del self.active_timers[name]
        return duration
    
    def report(self) -> None:
        if TIMING_MODE == 'suppressed' or not self.times:
            return
            
        longest_name = max(len(name) for name in self.times.keys())
        
        # Calculate max width needed for values
        all_durations = [duration for durations in self.times.values() for duration in durations]
        max_total = max(sum(self.times[name]) for name in self.times) if all_durations else 0
        max_width = max(len(f"{max_total:.6f}"), 10)
        
        separator = "=" * (longest_name + max_width * 2 + 30)
        header = f"\n{separator}\n"
        header += f"{'SCATTER TIMING REPORT':^{len(separator)}}\n"
        header += f"{separator}\n\n"
        
        header += f"{'SECTION':<{longest_name+2}} | {'TOTAL TIME':^{max_width+2}} | {'AVG TIME':^{max_width+2}} | {'CALLS':^8}\n"
        header += f"{'-'*(longest_name+2)}-+-{'-'*(max_width+2)}-+-{'-'*(max_width+2)}-+-{'-'*8}\n"
        mset.log(header)
        
        sorted_times = sorted(
            self.times.items(), 
            key=lambda x: sum(x[1]), 
            reverse=True
        )
        
        for name, durations in sorted_times:
            total = sum(durations)
            avg = total / len(durations) if durations else 0
            count = len(durations)
            mset.log(f"{name:<{longest_name+2}} | {total:>{max_width}.6f}s | {avg:>{max_width}.6f}s | {count:>8}\n")
        
        mset.log(f"\n{separator}\n")

timing_stats = TimingStats()

def timed(func=None, *, name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            timing_stats.start(timer_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                timing_stats.stop(timer_name)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

@contextmanager
def timed_section(name: str):
    timing_stats.start(name)
    try:
        yield
    finally:
        timing_stats.stop(name)


def normalize(vector: List[float]) -> List[float]:
    magnitude = math.sqrt(sum(comp**2 for comp in vector))
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return [comp / magnitude for comp in vector]


def cross_product(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def create_rotation_matrix(axis: List[float], angle: float) -> List[List[float]]:
    x, y, z = axis
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    one_minus_cos = 1 - cos_angle

    return [
        [
            cos_angle + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin_angle,
            x * z * one_minus_cos + y * sin_angle,
        ],
        [
            y * x * one_minus_cos + z * sin_angle,
            cos_angle + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin_angle,
        ],
        [
            z * x * one_minus_cos - y * sin_angle,
            z * y * one_minus_cos + x * sin_angle,
            cos_angle + z * z * one_minus_cos,
        ],
    ]


def convert_rotation_to_euler(matrix: List[List[float]]) -> List[float]:
    sy = math.sqrt(matrix[0][0] ** 2 + matrix[1][0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(matrix[2][1], matrix[2][2])
        y = math.atan2(-matrix[2][0], sy)
        z = math.atan2(matrix[1][0], matrix[0][0])
    else:
        x = math.atan2(-matrix[1][2], matrix[1][1])
        y = math.atan2(-matrix[2][0], sy)
        z = 0

    return [math.degrees(x), math.degrees(y), math.degrees(z)]


def convert_normal_to_rotation(normal: List[float]) -> List[float]:
    normal = normalize(normal)
    local_y = [0.0, 1.0, 0.0]
    cross = cross_product(local_y, normal)
    cross_magnitude = math.sqrt(sum(c**2 for c in cross))

    if cross_magnitude < 1e-6:
        return [0, 0, 0] if normal[1] > 0 else [180, 0, 0]

    axis = normalize(cross)
    dot = dot_product(local_y, normal)
    angle = math.atan2(cross_magnitude, dot)
    rotation_matrix = create_rotation_matrix(axis, angle)
    return convert_rotation_to_euler(rotation_matrix)


class ScatterTriangle:
    def __init__(self, indices, vertices, normals, uvs):
        self.indices = indices
        self.vertices = vertices
        self.normals = normals
        self.uvs = uvs

    @timed(name="triangle_area_calculation")
    def area(self) -> float:
        edge1 = [v2 - v1 for v1, v2 in zip(self.vertices[0], self.vertices[1])]
        edge2 = [v3 - v1 for v1, v3 in zip(self.vertices[0], self.vertices[2])]
        cross = cross_product(edge1, edge2)
        return 0.5 * math.sqrt(dot_product(cross, cross))

    @staticmethod
    def generate_random_barycentric() -> Tuple[float, float, float]:
        u, v = random.uniform(0, 1), random.uniform(0, 1)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        return 1 - u - v, u, v

    def calculate_barycentric_position(self, u: float, v: float, w: float) -> List[float]:
        return [self.vertices[0][i] * u + self.vertices[1][i] * v + self.vertices[2][i] * w for i in range(3)]

    def calculate_interpolated_normal(self, u: float, v: float, w: float) -> List[float]:
        interpolated = [self.normals[0][i] * u + self.normals[1][i] * v + self.normals[2][i] * w for i in range(3)]
        return normalize(interpolated)

    def calculate_interpolated_uv(self, u: float, v: float, w: float) -> List[float]:
        return [self.uvs[0][i] * u + self.uvs[1][i] * v + self.uvs[2][i] * w for i in range(2)]    


class ScatterPoint:
    def __init__(self, triangle, position, normal, barycentric, scale=None, rotation=None, mesh_object=None):
        self.triangle = triangle
        self.position = position
        self.normal = normal
        self.barycentric = barycentric
        self.uv = triangle.calculate_interpolated_uv(*barycentric)
        self.scale = scale or [1.0, 1.0, 1.0]
        self.rotation = [r + offset for r, offset in zip(convert_normal_to_rotation(normal), rotation or [0, 0, 0])]
        self.mesh_object = mesh_object

    def duplicate_mesh_object_to_point(self, name=None, apply_color=True) -> mset.MeshObject:
        object = self.mesh_object.duplicate()
        object.position = self.position
        object.rotation = self.rotation
        object.scale = self.scale
        if apply_color:
            rgba = [self.uv[0], self.uv[1], 0.0, 1.0]
            vertex_count = len(object.mesh.vertices) // 3
            color = []
            for _ in range(vertex_count):
                print(rgba)
                color.extend(rgba)
            object.mesh.colors = color
        if name:
            object.name = name
        return object


class ScatterMask:
    def __init__(self, mask_data, blend_method="multiply"):
        self.mask_data = mask_data
        self.blend_method = blend_method.lower()

    def get_value(self, u: float, v: float) -> float:
        if callable(self.mask_data):
            return self.mask_data(u, v)
        elif isinstance(self.mask_data, list):
            width, height = len(self.mask_data[0]), len(self.mask_data)
            x, y = int(u * width), int(v * height)
            return self.mask_data[y][x]
        else:
            raise TypeError(f"Unexpected mask_data type: {type(self.mask_data)}")


class ScatterSurface:
    def __init__(self, mesh_object: mset.MeshObject, seed: int = None) -> None:
        self.scene_object = mesh_object
        self.mesh = mesh_object.mesh
        self.vertices = self.mesh.vertices
        self.normals = self.mesh.normals
        self.uvs = self.mesh.uvs
        self.triangle_indices = self.mesh.triangles
        self.triangles: List[ScatterTriangle] = []
        self.cumulative_areas: List[float] = []
        self.scatter_masks: List[ScatterMask] = []
        self.scatter_points: List[ScatterPoint] = []
        self.scatter_mesh_objects: List[mset.MeshObject] = []
        self.seed = seed
        
        self._prepare_mesh_data()
        self._apply_random_seed(seed)

    @timed(name="prepare_mesh_data")
    def _prepare_mesh_data(self):
        cumulative_area = 0.0
        for i in range(0, len(self.triangle_indices), 3):
            indices = self.triangle_indices[i : i + 3]
            vertices = [self.vertices[idx * 3 : (idx + 1) * 3] for idx in indices]
            normals = [self.normals[idx * 3 : (idx + 1) * 3] for idx in indices]
            uvs = [self.uvs[idx * 2 : (idx + 1) * 2] for idx in indices]
            triangle = ScatterTriangle(indices, vertices, normals, uvs)
            cumulative_area += triangle.area()
            self.triangles.append(triangle)
            self.cumulative_areas.append(cumulative_area)

        if not self.triangles:
            raise ValueError("Mesh has no valid triangles for scattering.")

    @staticmethod
    def _apply_random_seed(seed: int) -> None:
        if seed is not None:
            random.seed(seed)

    def add_scatter_mask(self, mask_data, blend_method="multiply") -> ScatterMask:
        mask = ScatterMask(mask_data, blend_method)
        self.scatter_masks.append(mask)

    @timed(name="generate_scatter_point")
    def generate_scatter_point(self, mesh: mset.MeshObject = None) -> ScatterPoint:
        random_value = random.uniform(0, self.cumulative_areas[-1])
        index = bisect.bisect(self.cumulative_areas, random_value)
        triangle = self.triangles[index]

        u, v, w = triangle.generate_random_barycentric()
        
        with timed_section("apply_masks"):
            mask_value = self.apply_masks(u, v)
        
        if mask_value <= 0.0:
            return None

        position = triangle.calculate_barycentric_position(u, v, w)
        normal = triangle.calculate_interpolated_normal(u, v, w)
        scatter_point = ScatterPoint(triangle, position, normal, [u, v, w], mesh_object=mesh)
        self.scatter_points.append(scatter_point)
        return scatter_point

    def apply_masks(self, u: float, v: float) -> float:
        final_value = 1.0
        for mask in self.scatter_masks:
            mask_value = mask.get_value(u, v)
            if mask.blend_method == "multiply":
                final_value *= mask_value
            elif mask.blend_method == "add":
                final_value += mask_value
            elif mask.blend_method == "subtract":
                final_value -= mask_value
            else:
                raise ValueError(f"Unsupported blend method: {mask.blend_method}")
        return max(0.0, min(1.0, final_value))

    @timed(name="duplicate_mesh_objects")
    def duplicate_mesh_objects_to_points(self):
        for i, point in enumerate(self.scatter_points):
            self.scatter_mesh_objects.append(
                point.duplicate_mesh_object_to_point(name=f"ScatterPoint_{i}")
            )

class ScatterPlugin:
    def __init__(self) -> None:
        self.window: mset.UIWindow = mset.UIWindow("Scatter Tools")
        self.window.visible = True
        self._test()

    def _test(self, num_points: int = 1000) -> None:
        with timed_section("total_test"):
            selected_objects = mset.getSelectedObjects()
            if not selected_objects:
                mset.log("[Scatter Plugin] No objects selected. Exiting test.\n")
                return

            scatter_mesh = selected_objects[0]

            with timed_section("surface_creation"):
                scatter_surface = ScatterSurface(scatter_mesh)

            def checkerboard_mask(u, v, squares=2):
                return 1.0 if (int(u * squares) + int(v * squares)) % 2 == 0 else 0.0

            scatter_surface.add_scatter_mask(checkerboard_mask)

            # Save checkerboard mask visualization
            save_callable_mask_as_image(checkerboard_mask, Path("C:/Temp/checkerboard_mask.png"))

            # Save combined mask visualization
            save_combined_mask_image(scatter_surface, Path("C:/Temp/combined_mask.png"))

            # Initialize CSV log
            log_path = Path("C:/Temp/scatter_debug.csv")
            initialize_csv_logger(log_path)

            with timed_section("sphere_creation"):
                tris, verts, uvs, polys = uvsphere(0.05 / mset.getSceneUnitScale(), 10, 10)
                debug_mesh_data = mset.Mesh(triangles=tris, vertices=verts, uvs=uvs)
                debug_mesh_object = mset.MeshObject()
                debug_mesh_object.mesh = debug_mesh_data
                debug_mesh_object.addSubmesh(debug_mesh_object.name)
                debug_mesh_object.collapsed = True

            with timed_section("point_generation"):
                for _ in range(num_points):
                    random_value = random.uniform(0, scatter_surface.cumulative_areas[-1])
                    index = bisect.bisect(scatter_surface.cumulative_areas, random_value)
                    triangle = scatter_surface.triangles[index]
                    bary_u, bary_v, bary_w = triangle.generate_random_barycentric()
                    u, v = triangle.calculate_interpolated_uv(bary_u, bary_v, bary_w)
                    mask_value = scatter_surface.apply_masks(u, v)
                    accepted = mask_value > 0.0
                    log_csv_sample(log_path, u, v, mask_value, accepted, index)
                    if accepted:
                        point = ScatterPoint(triangle, triangle.calculate_barycentric_position(bary_u, bary_v, bary_w),
                                            triangle.calculate_interpolated_normal(bary_u, bary_v, bary_w),
                                            [bary_u, bary_v, bary_w], mesh_object=debug_mesh_object)
                        scatter_surface.scatter_points.append(point)

            scatter_surface.duplicate_mesh_objects_to_points()
            timing_stats.report()

        def _shutdown(self) -> None:
            mset.shutdownPlugin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scatter Plugin')
    parser.add_argument('--timing', choices=['suppressed', 'log', 'verbose'], 
                      default='log', help='Timing mode: suppressed (no timing), ' 
                      'log (custom timers only), or verbose (custom timers + cProfile)')
    
    args = parser.parse_args(None if len(sys.argv) > 1 else [])
    
    TIMING_MODE = args.timing
    
    if TIMING_MODE == 'verbose':
        with cProfile.Profile() as pr:
            ScatterPlugin()
        stats = Stats(pr)
        stats.strip_dirs()
        stats.sort_stats("cumtime")
        stats.print_stats()
    else:
        ScatterPlugin()