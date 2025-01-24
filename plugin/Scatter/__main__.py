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

debug_area_timer = 0.0
debug_parse_timer = 0.0
debug_vertices_timer = 0.0
debug_normals_timer = 0.0
debug_uvs_timer = 0.0
debug_random_triangle_timer = 0.0
debug_uvsphere_timer = 0.0
debug_prepare_mesh_timer = 0.0
debug_scattering_timer = 0.0
debug_total_timer = 0.0


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


def rotation_matrix_from_axis_angle(axis: List[float], angle: float) -> List[List[float]]:
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


def rotation_matrix_to_euler(matrix: List[List[float]]) -> List[float]:
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


def normal_to_rotation(normal: List[float]) -> List[float]:
    normal = normalize(normal)
    local_y = [0, 1, 0]
    cross = cross_product(local_y, normal)
    cross_magnitude = math.sqrt(sum(c**2 for c in cross))
    if cross_magnitude < 1e-6:
        return [0, 0, 0] if normal[1] > 0 else [180, 0, 0]
    axis = normalize(cross)
    dot = dot_product(local_y, normal)
    angle = math.atan2(cross_magnitude, dot)
    rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
    return rotation_matrix_to_euler(rotation_matrix)


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

        def area(self) -> float:
            global debug_area_timer
            area_start_time = time.time()
            edge1 = [self.vertices[1][i] - self.vertices[0][i] for i in range(3)]
            edge2 = [self.vertices[2][i] - self.vertices[0][i] for i in range(3)]
            cross = cross_product(edge1, edge2)
            area = 0.5 * math.sqrt(sum(c**2 for c in cross))
            debug_area_timer += time.time() - area_start_time
            return area

        def random_barycentric(self) -> Tuple[float, float, float]:
            u = random.random()
            v = random.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            return u, v, w

        def barycentric_position(self, u: float, v: float, w: float) -> List[float]:
            return [self.vertices[0][i] * u + self.vertices[1][i] * v + self.vertices[2][i] * w for i in range(3)]

        def interpolated_normal(self, u: float, v: float, w: float) -> List[float]:
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
            self.rotation = [normal_to_rotation(normal)[i] + rotation[i] for i in range(3)]
            self.mesh = mesh

        def duplicate_meshobject_to_xform(self) -> mset.MeshObject:
            if not self.mesh:
                (tris, verts, uvs, polys) = uvsphere(0.05 / mset.getSceneUnitScale(), 10, 10)
                debug_mesh_data = mset.Mesh(triangles=tris, vertices=verts, uvs=uvs)
                debug_mesh_object = mset.MeshObject()
                debug_mesh_object.mesh = debug_mesh_data
                debug_mesh_object.addSubmesh(debug_mesh_object.name)
                debug_mesh_object.collapsed = True
                self.mesh = debug_mesh_object
            mesh_object = self.mesh.duplicate()
            mesh_object.position = self.position
            mesh_object.rotation = self.rotation
            mesh_object.scale = self.scale
            return mesh_object

    def __init__(self, mesh_object: mset.MeshObject) -> None:
        self._scene_object = mesh_object
        self._mesh: mset.Mesh = mesh_object.mesh
        self._cumulative_areas: List[float] = list()
        self._triangles: List[ScatterSurface.ScatterTriangle] = list()
        self._vertices = self._mesh.vertices
        self._normals = self._mesh.normals
        self._uvs = self._mesh.uvs
        self._mesh_triangles = self._mesh.triangles
        self._prepare_mesh_data()
        self.scatter_points: List[self.ScatterPoint] = list()
        self.scatter_mesh_objects: List[mset.MeshObject] = list()

    def _parse_triangle(self, start_idx: int, vertices, normals, uvs, triangles) -> ScatterTriangle:
        global debug_parse_timer, debug_vertices_timer, debug_normals_timer, debug_uvs_timer
        parse_start = time.perf_counter()
        v1_idx, v2_idx, v3_idx = triangles[start_idx : start_idx + 3]
        vertices_start = time.perf_counter()
        verts = vertices[v1_idx * 3 : (v1_idx + 1) * 3], vertices[v2_idx * 3 : (v2_idx + 1) * 3], vertices[v3_idx * 3 : (v3_idx + 1) * 3]
        debug_vertices_timer += time.perf_counter() - vertices_start
        normals_start = time.perf_counter()
        norms = normals[v1_idx * 3 : (v1_idx + 1) * 3], normals[v2_idx * 3 : (v2_idx + 1) * 3], normals[v3_idx * 3 : (v3_idx + 1) * 3]
        debug_normals_timer += time.perf_counter() - normals_start
        uvs_start = time.perf_counter()
        uvs = uvs[v1_idx * 2 : (v1_idx + 1) * 2], uvs[v2_idx * 2 : (v2_idx + 1) * 2], uvs[v3_idx * 2 : (v3_idx + 1) * 2]
        debug_uvs_timer += time.perf_counter() - uvs_start
        debug_parse_timer += time.perf_counter() - parse_start
        return self.ScatterTriangle((v1_idx, v2_idx, v3_idx), verts, norms, uvs)

    def _add_triangle(self, triangle: ScatterTriangle, cumulative_area: float) -> float:
        area = triangle.area()
        self._cumulative_areas.append(cumulative_area + area)
        self._triangles.append(triangle)
        cumulative_area += area
        return cumulative_area

    def _prepare_mesh_data(self) -> None:
        global debug_prepare_mesh_timer
        prepare_start = time.perf_counter()
        mset.log(f"[Scatter Plugin] Preparing mesh data for {self._scene_object.name}... \n")
        cumulative_area = 0.0
        for i in range(0, len(self._mesh.triangles), 3):
            triangle = self._parse_triangle(i, self._vertices, self._normals, self._uvs, self._mesh_triangles)
            cumulative_area = self._add_triangle(triangle, cumulative_area)
        debug_prepare_mesh_timer += time.perf_counter() - prepare_start
        mset.log(f"[Scatter Plugin] Total parsing time: {debug_parse_timer:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Vertex extraction time: {debug_vertices_timer:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Normal extraction time: {debug_normals_timer:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - UV extraction time: {debug_uvs_timer:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Area calculation time: {debug_area_timer:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] Total prepare_mesh_data time: {debug_prepare_mesh_timer:.6f} seconds \n")
        mset.log("[Scatter Plugin] Mesh data preparation complete \n")
        mset.log("----------------------------------- \n")

    def random_triangle(self) -> ScatterTriangle:
        global debug_random_triangle_timer
        start_time = time.time()
        if not self._triangles:
            raise ValueError("No triangles available in the surface")
        random_value = random.uniform(0, self._cumulative_areas[-1])
        index = bisect.bisect(self._cumulative_areas, random_value)
        debug_random_triangle_timer += time.time() - start_time
        return self._triangles[index]

    def create_random_scatter_point(self) -> ScatterPoint:
        triangle = self.random_triangle()
        u, v, w = triangle.random_barycentric()
        position = triangle.barycentric_position(u, v, w)
        normal = triangle.interpolated_normal(u, v, w)
        point = self.ScatterPoint(triangle, position, normal, [u, v, w])
        self.scatter_points.append(point)
        return point

    def scatter_mesh_objects_to_xform(self) -> None:
        batch_size = 10
        scatter_batches = [self.scatter_points[i : i + batch_size] for i in range(0, len(self.scatter_points), batch_size)]

        def process_batch(batch):
            return [point.duplicate_meshobject_to_xform() for point in batch]

        with ThreadPoolExecutor() as executor:
            results = executor.map(process_batch, scatter_batches)
        for batch_result in results:
            self.scatter_mesh_objects.extend(batch_result)


class ScatterPlugin:
    def __init__(self) -> None:
        self.window: mset.UIWindow = mset.UIWindow("Scatter Tools")
        self.window.visible = True
        self._test()

    def _test(self) -> None:
        global debug_total_timer, debug_scattering_timer
        total_start = time.perf_counter()
        scatter_mesh = mset.getSelectedObjects()[0]
        scattering_start = time.perf_counter()
        scatter_surface = ScatterSurface(scatter_mesh)
        for i in range(100):
            scatter_surface.create_random_scatter_point()
        scatter_surface.scatter_mesh_objects_to_xform()
        debug_scattering_timer += time.perf_counter() - scattering_start
        debug_total_timer += time.perf_counter() - total_start
        mset.log(f"[Scatter Plugin] Total time: {debug_total_timer:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Preparing mesh data: {debug_prepare_mesh_timer:.6f} seconds \n")
        mset.log(f"[Scatter Plugin] - Scattering objects: {debug_scattering_timer:.6f} seconds \n")
        mset.log("----------------------------------- \n")

    def _shutdown(self) -> None:
        mset.shutdownPlugin()


if __name__ == "__main__":
    ScatterPlugin()
