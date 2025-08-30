import math

import mset
from procedural_masks import perlin_noise


def calculate_normals(vertices: list, triangles: list) -> list:
    vertex_count = len(vertices) // 3
    normals = [0.0] * len(vertices)
    
    for i in range(0, len(triangles), 3):
        i0, i1, i2 = triangles[i], triangles[i + 1], triangles[i + 2]
        
        v0 = [vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]]
        v1 = [vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]]
        v2 = [vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]]
        
        edge1 = [v1[j] - v0[j] for j in range(3)]
        edge2 = [v2[j] - v0[j] for j in range(3)]
        
        face_normal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        ]
        
        for idx in [i0, i1, i2]:
            normals[idx * 3] += face_normal[0]
            normals[idx * 3 + 1] += face_normal[1]
            normals[idx * 3 + 2] += face_normal[2]
    
    for i in range(vertex_count):
        nx = normals[i * 3]
        ny = normals[i * 3 + 1]
        nz = normals[i * 3 + 2]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        
        if length > 0:
            normals[i * 3] = nx / length
            normals[i * 3 + 1] = ny / length
            normals[i * 3 + 2] = nz / length
    
    return normals


def create_procedural_plane(width: float = 2.0, height: float = 2.0, subdivisions_x: int = 20, subdivisions_y: int = 20) -> mset.MeshObject:
    vertices = []
    triangles = []
    uvs = []
    
    unit_scale = mset.getSceneUnitScale()
    w = width / unit_scale
    h = height / unit_scale
    
    for y in range(subdivisions_y + 1):
        for x in range(subdivisions_x + 1):
            u = x / subdivisions_x
            v = y / subdivisions_y
            px = (u - 0.5) * w
            py = 0.0
            pz = (v - 0.5) * h
            vertices.extend([px, py, pz])
            uvs.extend([u, v])
    
    for y in range(subdivisions_y):
        for x in range(subdivisions_x):
            i0 = y * (subdivisions_x + 1) + x
            i1 = i0 + 1
            i2 = (y + 1) * (subdivisions_x + 1) + x
            i3 = i2 + 1
            triangles.extend([i0, i2, i1])
            triangles.extend([i1, i2, i3])
    
    mesh = mset.Mesh(triangles=triangles, vertices=vertices, uvs=uvs)
    obj = mset.MeshObject()
    obj.mesh = mesh
    obj.name = "ScatterPlane"
    obj.addSubmesh(obj.name)
    return obj


def create_terrain(width: float = 2.0, height: float = 2.0, subdivisions_x: int = 50, subdivisions_y: int = 50, 
                  noise_scale: float = 0.3, noise_frequency: float = 4.0) -> mset.MeshObject:
    vertices = []
    triangles = []
    uvs = []
    
    unit_scale = mset.getSceneUnitScale()
    w = width / unit_scale
    h = height / unit_scale
    height_scale = noise_scale / unit_scale
    
    for y in range(subdivisions_y + 1):
        for x in range(subdivisions_x + 1):
            u = x / subdivisions_x
            v = y / subdivisions_y
            px = (u - 0.5) * w
            pz = (v - 0.5) * h
            
            noise_height = perlin_noise(u, v, noise_frequency)
            py = (noise_height - 0.5) * height_scale
            
            vertices.extend([px, py, pz])
            uvs.extend([u, v])
    
    for y in range(subdivisions_y):
        for x in range(subdivisions_x):
            i0 = y * (subdivisions_x + 1) + x
            i1 = i0 + 1
            i2 = (y + 1) * (subdivisions_x + 1) + x
            i3 = i2 + 1
            triangles.extend([i0, i2, i1])
            triangles.extend([i1, i2, i3])
    
    normals = calculate_normals(vertices, triangles)
    
    mesh = mset.Mesh(triangles=triangles, vertices=vertices, uvs=uvs, normals=normals)
    obj = mset.MeshObject()
    obj.mesh = mesh
    obj.name = "ScatterTerrain"
    obj.addSubmesh(obj.name)
    return obj


def create_uv_sphere(radius: float = 1.0, rings: int = 20, segments: int = 40) -> mset.MeshObject:
    vertices = []
    triangles = []
    uvs = []
    
    unit_scale = mset.getSceneUnitScale()
    r = radius / unit_scale
    
    for ring in range(rings + 1):
        theta = ring * math.pi / rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        for segment in range(segments + 1):
            phi = segment * 2.0 * math.pi / segments
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
            x = r * sin_theta * cos_phi
            y = r * cos_theta
            z = r * sin_theta * sin_phi
            
            vertices.extend([x, y, z])
            
            u = segment / segments
            v = ring / rings
            uvs.extend([u, v])
    
    for ring in range(rings):
        for segment in range(segments):
            current = ring * (segments + 1) + segment
            next_ring = (ring + 1) * (segments + 1) + segment
            
            triangles.extend([current, current + 1, next_ring])
            triangles.extend([current + 1, next_ring + 1, next_ring])
    
    mesh = mset.Mesh(triangles=triangles, vertices=vertices, uvs=uvs)
    obj = mset.MeshObject()
    obj.mesh = mesh
    obj.name = "ScatterSphere"
    obj.addSubmesh(obj.name)
    return obj