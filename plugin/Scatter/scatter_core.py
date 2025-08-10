import bisect
import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import mset

generate_primitives_path = Path("C:\\Program Files\\Marmoset\\Toolbag 5\\data\\plugin\\Generate Primitives")
sys.path.append(str(generate_primitives_path))
module_path = generate_primitives_path / "uvsphere.py"
spec = importlib.util.spec_from_file_location("uvsphere", module_path)
uvsphere_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uvsphere_module)
uvsphere = uvsphere_module.uvsphere


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
            return float(self.mask_data(u, v))
        elif isinstance(self.mask_data, list):
            height = len(self.mask_data)
            width = len(self.mask_data[0]) if height > 0 else 0
            x = max(0, min(width - 1, int(u * width)))
            y = max(0, min(height - 1, int(v * height)))
            return float(self.mask_data[y][x])
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
        return mask

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

    def generate_scatter_point(self, mesh: mset.MeshObject = None):
        random_value = random.uniform(0, self.cumulative_areas[-1])
        index = bisect.bisect(self.cumulative_areas, random_value)
        triangle = self.triangles[index]
        bary_u, bary_v, bary_w = triangle.generate_random_barycentric()
        uv_u, uv_v = triangle.calculate_interpolated_uv(bary_u, bary_v, bary_w)
        mask_value = self.apply_masks(uv_u, uv_v)
        if mask_value <= 0.0:
            return None
        position = triangle.calculate_barycentric_position(bary_u, bary_v, bary_w)
        normal = triangle.calculate_interpolated_normal(bary_u, bary_v, bary_w)
        scatter_point = ScatterPoint(triangle, position, normal, [bary_u, bary_v, bary_w], mesh_object=mesh)
        self.scatter_points.append(scatter_point)
        return scatter_point

    def duplicate_mesh_objects_to_points(self):
        for i, point in enumerate(self.scatter_points):
            self.scatter_mesh_objects.append(point.duplicate_mesh_object_to_point(name=f"ScatterPoint_{i}"))


def build_uvsphere_instance() -> mset.MeshObject:
    tris, verts, uvs, polys = uvsphere(0.05 / mset.getSceneUnitScale(), 10, 10)
    mesh = mset.Mesh(triangles=tris, vertices=verts, uvs=uvs)
    obj = mset.MeshObject()
    obj.mesh = mesh
    obj.addSubmesh(obj.name)
    obj.collapsed = True
    return obj


def scatter_with_primitive(num_points: int = 1000, seed: int | None = None) -> None:
    selected = mset.getSelectedObjects()
    if not selected:
        mset.log("No objects selected.")
        return
    scatter_mesh = selected[0]
    surface = ScatterSurface(scatter_mesh, seed=seed)
    instance = build_uvsphere_instance()
    for _ in range(num_points):
        surface.generate_scatter_point(mesh=instance)
    surface.duplicate_mesh_objects_to_points()


def scatter_with_selected_prototype(num_points: int = 1000, seed: int | None = None) -> None:
    selected = mset.getSelectedObjects()
    if not selected or len(selected) < 2:
        mset.log("Select the scatter surface first, then the prototype object.")
        return
    surface_obj, proto_obj = selected[0], selected[1]
    surface = ScatterSurface(surface_obj, seed=seed)
    for _ in range(num_points):
        surface.generate_scatter_point(mesh=proto_obj)
    surface.duplicate_mesh_objects_to_points()
