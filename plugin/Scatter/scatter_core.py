import bisect
import math
import random
from typing import List, Tuple

import mset
from math_utils import convert_normal_to_rotation, cross_product, dot_product, normalize
from procedural_meshes import create_procedural_plane, create_terrain, create_uv_sphere


def create_scene_and_scatter(surface_type: str, num_points: int = 1000, seed: int | None = None) -> None:
    try:
        if surface_type == "plane":
            scatter_surface = create_procedural_plane()
        elif surface_type == "sphere":
            scatter_surface = create_uv_sphere()
        elif surface_type == "terrain":
            scatter_surface = create_terrain()
        else:
            mset.err(f"Unknown surface type: {surface_type}")
            return

        mset.log(f"Created {surface_type} scatter surface\n")

        surface = ScatterSurface(scatter_surface, seed=seed)
        instance = create_uv_sphere(0.05, 6, 12)
        instance.name = "ScatterSphere"
        instance.collapsed = True

        for _ in range(num_points):
            surface.generate_scatter_point(mesh=instance)

        surface.duplicate_mesh_objects_to_points()

        actual_points = len(surface.scatter_points)
        mset.log(f"Successfully scattered {actual_points} spheres on {surface_type} surface\n")

    except Exception as e:
        mset.err(f"Failed to create scene and scatter: {str(e)}")


def scatter_with_primitive(num_points: int = 1000, seed: int | None = None) -> None:
    selected = mset.getSelectedObjects()
    if not selected:
        mset.err("Please select a mesh object to scatter on.")
        return

    scatter_mesh = selected[0]
    if not isinstance(scatter_mesh, mset.MeshObject):
        mset.err("Selected object is not a mesh. Please select a mesh object.")
        return

    mset.log(f"Scattering {num_points} spheres on {scatter_mesh.name}...\n")

    try:
        surface = ScatterSurface(scatter_mesh, seed=seed)
        instance = create_uv_sphere(0.05, 6, 12)
        instance.name = "ScatterSphere"
        instance.collapsed = True

        for _ in range(num_points):
            surface.generate_scatter_point(mesh=instance)

        surface.duplicate_mesh_objects_to_points()

        actual_points = len(surface.scatter_points)
        mset.log(f"Successfully scattered {actual_points} spheres on {scatter_mesh.name}\n")

    except Exception as e:
        mset.err(f"Failed to scatter objects: {str(e)}")


def scatter_with_selected_prototype(num_points: int = 1000, seed: int | None = None) -> None:
    selected = mset.getSelectedObjects()
    if not selected or len(selected) < 2:
        mset.log("Select the scatter surface first, then the prototype object.\n")
        return
    surface_obj, proto_obj = selected[0], selected[1]
    surface = ScatterSurface(surface_obj, seed=seed)
    for _ in range(num_points):
        surface.generate_scatter_point(mesh=proto_obj)
    surface.duplicate_mesh_objects_to_points()


class ScatterMask:
    def __init__(self, mask_data, blend_method="multiply"):
        self.blend_method = blend_method.lower()
        self.mask_data = mask_data

    def get_value(self, u: float, v: float) -> float:
        if callable(self.mask_data):
            result = self.mask_data(u, v)
            return float(result) if result is not None else 0.0
        elif isinstance(self.mask_data, list):
            height = len(self.mask_data)
            width = len(self.mask_data[0]) if height > 0 else 0
            x = max(0, min(width - 1, int(u * width)))
            y = max(0, min(height - 1, int(v * height)))
            return float(self.mask_data[y][x])
        else:
            raise TypeError(f"Unexpected mask_data type: {type(self.mask_data)}")


class ScatterPoint:
    def __init__(self, triangle, position, normal, barycentric, scale=None, rotation=None, mesh_object=None):
        self.barycentric = barycentric
        self.mesh_object = mesh_object
        self.normal = normal
        self.position = position
        self.rotation = [r + offset for r, offset in zip(convert_normal_to_rotation(normal), rotation or [0, 0, 0])]
        self.scale = scale or [1.0, 1.0, 1.0]
        self.triangle = triangle
        self.uv = triangle.calculate_interpolated_uv(*barycentric)

    def duplicate_mesh_object_to_point(self, name=None, apply_color=True) -> mset.MeshObject:
        if self.mesh_object is None:
            raise ValueError("No mesh object available for duplication")
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


class ScatterSurface:
    def __init__(self, mesh_object: mset.MeshObject, seed: int | None = None) -> None:
        self.cumulative_areas: List[float] = []
        self.mesh = mesh_object.mesh
        self.normals = self.mesh.normals
        self.scatter_masks: List[ScatterMask] = []
        self.scatter_mesh_objects: List[mset.MeshObject] = []
        self.scatter_points: List[ScatterPoint] = []
        self.scene_object = mesh_object
        self.seed = seed
        self.triangle_indices = self.mesh.triangles
        self.triangles: List[ScatterTriangle] = []
        self.uvs = self.mesh.uvs
        self.vertices = self.mesh.vertices
        self._prepare_mesh_data()
        self._apply_random_seed(seed)

    def _apply_random_seed(self, seed: int | None) -> None:
        if seed is not None:
            random.seed(seed)

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

    def duplicate_mesh_objects_to_points(self):
        for i, point in enumerate(self.scatter_points):
            self.scatter_mesh_objects.append(point.duplicate_mesh_object_to_point(name=f"ScatterPoint_{i}"))

    def generate_scatter_point(self, mesh: mset.MeshObject | None = None):
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


class ScatterTriangle:
    def __init__(self, indices, vertices, normals, uvs):
        self.indices = indices
        self.normals = normals
        self.uvs = uvs
        self.vertices = vertices

    @staticmethod
    def generate_random_barycentric() -> Tuple[float, float, float]:
        u, v = random.uniform(0, 1), random.uniform(0, 1)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        return 1 - u - v, u, v

    def area(self) -> float:
        edge1 = [v2 - v1 for v1, v2 in zip(self.vertices[0], self.vertices[1])]
        edge2 = [v3 - v1 for v1, v3 in zip(self.vertices[0], self.vertices[2])]
        cross = cross_product(edge1, edge2)
        return 0.5 * math.sqrt(dot_product(cross, cross))

    def calculate_barycentric_position(self, u: float, v: float, w: float) -> List[float]:
        return [self.vertices[0][i] * u + self.vertices[1][i] * v + self.vertices[2][i] * w for i in range(3)]

    def calculate_interpolated_normal(self, u: float, v: float, w: float) -> List[float]:
        interpolated = [self.normals[0][i] * u + self.normals[1][i] * v + self.normals[2][i] * w for i in range(3)]
        return normalize(interpolated)

    def calculate_interpolated_uv(self, u: float, v: float, w: float) -> List[float]:
        return [self.uvs[0][i] * u + self.uvs[1][i] * v + self.uvs[2][i] * w for i in range(2)]
