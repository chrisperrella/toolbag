import sys
import importlib.util
import random
import math
import bisect
import mset
from typing import List, Tuple
from pathlib import Path

generate_primitives_path = Path("C:\Program Files\Marmoset\Toolbag 5\data\plugin\Generate Primitives")
sys.path.append(str(generate_primitives_path))
module_path = generate_primitives_path / "uvsphere.py"
spec = importlib.util.spec_from_file_location("uvsphere", module_path)
uvsphere_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uvsphere_module)
uvsphere = uvsphere_module.uvsphere

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
    cross_magnitude = math.sqrt(sum(c ** 2 for c in cross))
    if cross_magnitude < 1e-6:
        return [0, 0, 0] if normal[1] > 0 else [180, 0, 0]
    axis = normalize(cross)
    dot = dot_product(local_y, normal)
    angle = math.atan2(cross_magnitude, dot)
    rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)
    return rotation_matrix_to_euler(rotation_matrix)

def normalize(vector: List[float]) -> List[float]:
    magnitude = math.sqrt(sum(comp ** 2 for comp in vector))
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
            edge1 = [self.vertices[1][i] - self.vertices[0][i] for i in range(3)]
            edge2 = [self.vertices[2][i] - self.vertices[0][i] for i in range(3)]
            cross = cross_product(edge1, edge2)
            return 0.5 * math.sqrt(dot_product(cross, cross))

        def random_barycentric(self) -> Tuple[float, float, float]:
            u = random.random()
            v = random.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            w = 1 - u - v
            return u, v, w

        def barycentric_position(self, u: float, v: float, w: float) -> List[float]:
            return [
                self.vertices[0][i] * u + self.vertices[1][i] * v + self.vertices[2][i] * w
                for i in range(3)
            ]

        def interpolated_normal(self, u: float, v: float, w: float) -> List[float]:
            interpolated = [
                self.normals[0][i] * u + self.normals[1][i] * v + self.normals[2][i] * w
                for i in range(3)
            ]
            return normalize(interpolated)

    def __init__(self, mesh_object: mset.MeshObject) -> None:
        self._mesh: mset.Mesh = mesh_object.mesh
        self._cumulative_areas: List[float] = []
        self._triangles: List[ScatterSurface.ScatterTriangle] = []
        self._prepare_mesh_data()

    def _parse_triangle(self, start_idx: int) -> ScatterTriangle:
        v1_idx, v2_idx, v3_idx = self._mesh.triangles[start_idx:start_idx + 3]
        vertices = (
            self._mesh.vertices[v1_idx * 3:v1_idx * 3 + 3],
            self._mesh.vertices[v2_idx * 3:v2_idx * 3 + 3],
            self._mesh.vertices[v3_idx * 3:v3_idx * 3 + 3],
        )
        normals = (
            self._mesh.normals[v1_idx * 3:v1_idx * 3 + 3],
            self._mesh.normals[v2_idx * 3:v2_idx * 3 + 3],
            self._mesh.normals[v3_idx * 3:v3_idx * 3 + 3],
        )
        uvs = (
            self._mesh.uvs[v1_idx * 2:v1_idx * 2 + 2],
            self._mesh.uvs[v2_idx * 2:v2_idx * 2 + 2],
            self._mesh.uvs[v3_idx * 2:v3_idx * 2 + 2],
        )
        return self.ScatterTriangle((v1_idx, v2_idx, v3_idx), vertices, normals, uvs)

    def _add_triangle(self, triangle: ScatterTriangle, cumulative_area: float) -> float:
        area = triangle.area()
        self._cumulative_areas.append(cumulative_area + area)
        self._triangles.append(triangle)
        return cumulative_area + area

    def _prepare_mesh_data(self) -> None:
        cumulative_area = 0.0
        for i in range(0, len(self._mesh.triangles), 3):
            triangle = self._parse_triangle(i)
            cumulative_area = self._add_triangle(triangle, cumulative_area)

    def random_triangle(self) -> ScatterTriangle:
        if not self._triangles:
            raise ValueError("No triangles available in the surface")
        random_value = random.uniform(0, self._cumulative_areas[-1])
        index = bisect.bisect(self._cumulative_areas, random_value)
        return self._triangles[index]

class ScatterPlugin:
    def __init__(self) -> None:
        self.window: mset.UIWindow = mset.UIWindow("Scatter Tools")
        self.window.visible = True
        self._test()

    def _get_random_transform_from_scatter_surface(self, scatter_surface: ScatterSurface) -> Tuple[List[float], List[float]]:
        triangle = scatter_surface.random_triangle()
        u, v, w = triangle.random_barycentric()
        position = triangle.barycentric_position(u, v, w)
        normal = triangle.interpolated_normal(u, v, w)
        rotation = normal_to_rotation(normal)
        return position, rotation

    def _test(self) -> None:
        mesh_object = mset.findObject("Sphere")
        scatter_surface = ScatterSurface(mesh_object)
        for i in range(100):            
            position, rotation = self._get_random_transform_from_scatter_surface(scatter_surface)
            (tris, verts, uvs, polys) = uvsphere(0.05 / mset.getSceneUnitScale(), 10, 10)
            scene_prim_mesh = mset.Mesh(triangles=tris, vertices=verts, uvs=uvs)
            scene_prim = mset.MeshObject()
            scene_prim.mesh = scene_prim_mesh
            scene_prim.addSubmesh(scene_prim.name)
            scene_prim.collapsed = True
            scene_prim.position = position
            scene_prim.rotation = rotation

    def _shutdown(self) -> None:
        mset.shutdownPlugin()

if __name__ == "__main__":
    ScatterPlugin()