import bisect
import cProfile
import importlib.util
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pstats import Stats
from typing import List, Tuple

import mset

generate_primitives_path = Path("C:\\Program Files\\Marmoset\\Toolbag 5\\data\\plugin\\Generate Primitives")
sys.path.append(str(generate_primitives_path))
module_path = generate_primitives_path / "uvsphere.py"
spec = importlib.util.spec_from_file_location("uvsphere", module_path)
uvsphere_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uvsphere_module)
uvsphere = uvsphere_module.uvsphere


class DebugTimer:
    def __init__(self):
        self.timers = {
            "debug_area_timer": 0.0,
            "debug_parse_timer": 0.0,
            "debug_random_triangle_timer": 0.0,
            "debug_prepare_mesh_timer": 0.0,
            "debug_scattering_timer": 0.0,
            "debug_mask_timer": 0.0,
            "debug_triangle_data_timer": 0.0,
        }

    def add_time(self, name: str, duration: float) -> None:
        if name not in self.timers:
            raise ValueError(f"Invalid timer name: {name}")
        self.timers[name] += duration

    def get_timer(self, name: str) -> float:
        return self.timers.get(name, 0.0)

    def calculate_total_time(self) -> float:
        """Calculate total time as the sum of all individual timers."""
        return sum(self.timers.values())

    def log_timers(self) -> None:
        total_time = self.calculate_total_time()
        mset.log(f"[Scatter Plugin] Total time: {total_time:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Preparing mesh data: {self.get_timer('debug_prepare_mesh_timer'):.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Triangle data extraction: {self.get_timer('debug_triangle_data_timer'):.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Area calculation: {self.get_timer('debug_area_timer'):.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Random triangle selection: {self.get_timer('debug_random_triangle_timer'):.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Mask application: {self.get_timer('debug_mask_timer'):.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Scattering objects: {self.get_timer('debug_scattering_timer'):.6f} seconds \n")
        mset.log("----------------------------------- \n")

    def timer(self, timer_name: str) -> callable:
        def decorator(func: callable) -> callable:
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                self.add_time(timer_name, duration)
                return result

            return wrapper

        return decorator


timers = DebugTimer()


def cross_product(a: List[float], b: List[float]) -> List[float]:
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Both vectors must be 3-dimensional")
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot_product(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must be of the same dimension")
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
    local_y = [0, 1, 0]
    cross = cross_product(local_y, normal)
    cross_magnitude = math.sqrt(sum(c**2 for c in cross))
    if cross_magnitude < 1e-6:
        return [0, 0, 0] if normal[1] > 0 else [180, 0, 0]
    axis = normalize(cross)
    dot = dot_product(local_y, normal)
    angle = math.atan2(cross_magnitude, dot)
    rotation_matrix = create_rotation_matrix(axis, angle)
    return convert_rotation_to_euler(rotation_matrix)


def normalize(vector: List[float]) -> List[float]:
    magnitude = math.sqrt(sum(comp**2 for comp in vector))
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return [comp / magnitude for comp in vector]


class ScatterSurface:
    class ScatterTriangle:
        def __init__(
            self,
            indices: Tuple[int, int, int],
            vertices: Tuple[List[float], List[float], List[float]],
            normals: Tuple[List[float], List[float], List[float]],
            uvs: Tuple[List[float], List[float], List[float]],
        ) -> None:
            self.indices = indices
            self.vertices = vertices
            self.normals = normals
            self.uvs = uvs

        @timers.timer("debug_area_timer")
        def area(self) -> float:
            edge1 = [self.vertices[1][i] - self.vertices[0][i] for i in range(3)]
            edge2 = [self.vertices[2][i] - self.vertices[0][i] for i in range(3)]
            cross = cross_product(edge1, edge2)
            area = 0.5 * math.sqrt(sum(c**2 for c in cross))
            return area

        @staticmethod
        def generate_random_barycentric() -> Tuple[float, float, float]:
            u = random.random()
            v = random.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            return u, v, w

        def calculate_barycentric_position(self, u: float, v: float, w: float) -> List[float]:
            return [self.vertices[0][i] * u + self.vertices[1][i] * v + self.vertices[2][i] * w for i in range(3)]

        def calculate_interpolated_normal(self, u: float, v: float, w: float) -> List[float]:
            interpolated = [self.normals[0][i] * u + self.normals[1][i] * v + self.normals[2][i] * w for i in range(3)]
            return normalize(interpolated)

    class ScatterPoint:
        def __init__(
            self,
            triangle: "ScatterSurface.ScatterTriangle",
            position: List[float],
            normal: List[float],
            barycentric: Tuple[float, float, float],
            scale: List[float] = [1.0, 1.0, 1.0],
            rotation: List[float] = [0, 0, 0],
            mesh: mset.MeshObject = None,
        ) -> None:
            self.triangle = triangle
            self.position = position
            self.normal = normal
            self.barycentric = barycentric
            self.scale = scale
            self.rotation = [convert_normal_to_rotation(normal)[i] + rotation[i] for i in range(3)]
            self.source_mesh = mesh

        def duplicate_mesh_object_to_point(self, name: str = None) -> mset.MeshObject:
            mesh_object = self.source_mesh.duplicate()
            mesh_object.position = self.position
            mesh_object.rotation = self.rotation
            mesh_object.scale = self.scale
            if name:
                mesh_object.name = name
            return mesh_object

    class ScatterMask:
        def __init__(self, mask_data, blend_method="multiply"):
            self.mask_data = mask_data
            self.blend_method = blend_method

        def get_value(self, u: float, v: float) -> float:
            if callable(self.mask_data):
                result = self.mask_data(u, v)
                return result
            elif isinstance(self.mask_data, list):
                width, height = len(self.mask_data[0]), len(self.mask_data)
                x, y = int(u * width), int(v * height)
                result = self.mask_data[y][x]
                return result
            else:
                raise TypeError(f"Unexpected mask_data type: {type(self.mask_data)}")

    def __init__(self, mesh_object: mset.MeshObject, seed: int = None) -> None:
        self._scene_object = mesh_object
        self._mesh: mset.Mesh = mesh_object.mesh
        self._triangles: List[ScatterSurface.ScatterTriangle] = list()
        self._cumulative_areas: List[float] = list()
        self._vertices = self._mesh.vertices
        self._normals = self._mesh.normals
        self._uvs = self._mesh.uvs
        self._mesh_triangles = self._mesh.triangles
        self.masks: List[ScatterSurface.ScatterMask] = list()
        self.scatter_points: List[self.ScatterPoint] = list()
        self.scatter_mesh_objects: List[mset.MeshObject] = list()
        self.seed = seed
        self._prepare_mesh_data()
        self._apply_random_seed(self.seed)

    @staticmethod
    def _apply_random_seed(seed: int) -> None:
        if seed is not None:
            random.seed(seed)

    @timers.timer("debug_triangle_data_timer")
    def _extract_triangle_data(self, start_idx: int, vertices, normals, uvs, triangles) -> ScatterTriangle:
        v1_idx, v2_idx, v3_idx = triangles[start_idx : start_idx + 3]
        verts = vertices[v1_idx * 3 : (v1_idx + 1) * 3], vertices[v2_idx * 3 : (v2_idx + 1) * 3], vertices[v3_idx * 3 : (v3_idx + 1) * 3]
        norms = normals[v1_idx * 3 : (v1_idx + 1) * 3], normals[v2_idx * 3 : (v2_idx + 1) * 3], normals[v3_idx * 3 : (v3_idx + 1) * 3]
        uvs = uvs[v1_idx * 2 : (v1_idx + 1) * 2], uvs[v2_idx * 2 : (v2_idx + 1) * 2], uvs[v3_idx * 2 : (v3_idx + 1) * 2]
        return self.ScatterTriangle((v1_idx, v2_idx, v3_idx), verts, norms, uvs)

    def _add_triangle_with_area(self, triangle: ScatterTriangle, cumulative_area: float) -> float:
        area = triangle.area()
        self._cumulative_areas.append(cumulative_area + area)
        self._triangles.append(triangle)
        cumulative_area += area
        return cumulative_area

    @timers.timer("debug_prepare_mesh_timer")
    def _prepare_mesh_data(self) -> None:
        mset.log(f"[Scatter Plugin] Preparing mesh data for {self._scene_object.name}... \n")
        cumulative_area = 0.0
        for i in range(0, len(self._mesh.triangles), 3):
            triangle = self._extract_triangle_data(i, self._vertices, self._normals, self._uvs, self._mesh_triangles)
            cumulative_area = self._add_triangle_with_area(triangle, cumulative_area)

    @timers.timer("debug_random_triangle_timer")
    def select_random_triangle(self) -> ScatterTriangle:
        if not self._triangles:
            raise ValueError("No triangles available in the surface")
        random_value = random.uniform(0, self._cumulative_areas[-1])
        index = bisect.bisect(self._cumulative_areas, random_value)
        return self._triangles[index]

    @timers.timer("debug_scattering_timer")
    def generate_scatter_point(self, mesh: mset.MeshObject = None) -> ScatterPoint:
        triangle = self.select_random_triangle()
        u, v, w = triangle.generate_random_barycentric()
        mask_value = self.apply_masks(u, v)
        if mask_value <= 0.0:
            return None
        position = triangle.calculate_barycentric_position(u, v, w)
        normal = triangle.calculate_interpolated_normal(u, v, w)
        point = self.ScatterPoint(triangle, position, normal, [u, v, w], mesh=mesh)
        self.scatter_points.append(point)
        return point

    def duplicate_mesh_objects_to_points(self) -> None:
        batch_size = 10
        scatter_batches = [self.scatter_points[i : i + batch_size] for i in range(0, len(self.scatter_points), batch_size)]

        def process_batch(batch, start_index):
            return [point.duplicate_mesh_object_to_point(name=f"ScatterPoint_{start_index + idx}") for idx, point in enumerate(batch)]

        with ThreadPoolExecutor() as executor:
            results = [executor.submit(process_batch, batch, start_index) for start_index, batch in enumerate(scatter_batches, start=0)]

            for future in results:
                self.scatter_mesh_objects.extend(future.result())

    def add_scatter_mask(self, mask_data, blend_method="multiply") -> None:
        mask = self.ScatterMask(mask_data, blend_method)
        self.masks.append(mask)

    @timers.timer("debug_mask_timer")
    def apply_masks(self, u: float, v: float) -> float:
        final_value = 1.0
        for mask in self.masks:
            mask_value = mask.get_value(u, v)
            if mask.blend_method == "multiply":
                final_value *= mask_value
            elif mask.blend_method == "add":
                final_value += mask_value
            elif mask.blend_method == "subtract":
                final_value -= mask_value
            final_value = max(0.0, min(1.0, final_value))
        return final_value


class ScatterPlugin:
    def __init__(self) -> None:
        self.window: mset.UIWindow = mset.UIWindow("Scatter Tools")
        self.window.visible = True
        self._test()

    def _test(self) -> None:
        def gradient_mask(u, v):
            return u * v

        def checkerboard_mask(u, v, num_squares=6):
            scaled_u = int(u * num_squares)
            scaled_v = int(v * num_squares)
            if (scaled_u + scaled_v) % 2 == 0:
                return 1.0
            else:
                return 0.0

        selected_objects = mset.getSelectedObjects()
        if not selected_objects:
            mset.log("[Scatter Plugin] No objects selected. Exiting test.\n")
            return

        scatter_mesh = selected_objects[0]
        scatter_surface = ScatterSurface(scatter_mesh)

        scatter_surface.add_scatter_mask(lambda u, v: checkerboard_mask(u, v, num_squares=6), blend_method="multiply")

        (tris, verts, uvs, polys) = uvsphere(0.05 / mset.getSceneUnitScale(), 10, 10)
        debug_mesh_data = mset.Mesh(triangles=tris, vertices=verts, uvs=uvs)
        debug_mesh_object = mset.MeshObject()
        debug_mesh_object.mesh = debug_mesh_data
        debug_mesh_object.addSubmesh(debug_mesh_object.name)
        debug_mesh_object.collapsed = True

        for i in range(1000):
            scatter_surface.generate_scatter_point(debug_mesh_object)

        scatter_surface.duplicate_mesh_objects_to_points()
        timers.log_timers()

    def _shutdown(self) -> None:
        mset.shutdownPlugin()


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        ScatterPlugin()
    stats = Stats(pr)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats()
