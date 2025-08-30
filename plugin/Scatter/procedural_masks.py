import math
from typing import Tuple

from math_utils import fade, grad, hash_2d, lerp


def cellular_noise(u: float, v: float, frequency: float = 8.0, threshold: float = 0.5) -> float:
    return 1.0 if voronoi_noise(u, v, frequency) > threshold else 0.0


def checkerboard(u: float, v: float, squares: int = 2) -> float:
    return 1.0 if (int(u * squares) + int(v * squares)) % 2 == 0 else 0.0


def circle(u: float, v: float, center: Tuple[float, float] = (0.5, 0.5), radius: float = 0.3) -> float:
    cx, cy = center
    du, dv = u - cx, v - cy
    distance = math.sqrt(du * du + dv * dv)
    return 1.0 if distance <= radius else 0.0


def fbm_noise(u: float, v: float, octaves: int = 4, frequency: float = 4.0, 
             amplitude: float = 1.0, lacunarity: float = 2.0, gain: float = 0.5) -> float:
    value = 0.0
    freq = frequency
    amp = amplitude
    
    for _ in range(octaves):
        value += perlin_noise(u, v, freq) * amp
        freq *= lacunarity
        amp *= gain
    
    return min(1.0, max(0.0, value))


def gradient_horizontal(u: float, v: float) -> float:
    return u


def gradient_vertical(u: float, v: float) -> float:
    return v


def noise_simple(u: float, v: float, frequency: float = 8.0) -> float:
    x = u * frequency
    y = v * frequency
    return (math.sin(x) * math.sin(y) + 1.0) * 0.5


def perlin_noise(u: float, v: float, frequency: float = 8.0) -> float:
    x = u * frequency
    y = v * frequency
    
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    sx = fade(x - x0)
    sy = fade(y - y0)
    
    n0 = grad(hash_2d(x0, y0), x - x0, y - y0)
    n1 = grad(hash_2d(x1, y0), x - x1, y - y0)
    ix0 = lerp(n0, n1, sx)
    
    n0 = grad(hash_2d(x0, y1), x - x0, y - y1)
    n1 = grad(hash_2d(x1, y1), x - x1, y - y1)
    ix1 = lerp(n0, n1, sx)
    
    value = lerp(ix0, ix1, sy)
    return (value + 1.0) * 0.5


def radial_gradient(u: float, v: float, center: Tuple[float, float] = (0.5, 0.5)) -> float:
    cx, cy = center
    du, dv = u - cx, v - cy
    distance = math.sqrt(du * du + dv * dv)
    return min(1.0, distance * 2.0)


def ridged_noise(u: float, v: float, octaves: int = 4, frequency: float = 4.0) -> float:
    value = 0.0
    freq = frequency
    amp = 1.0
    
    for _ in range(octaves):
        n = perlin_noise(u, v, freq)
        n = 1.0 - abs(n - 0.5) * 2.0
        value += n * amp
        freq *= 2.0
        amp *= 0.5
    
    return min(1.0, max(0.0, value))


def ring(u: float, v: float, center: Tuple[float, float] = (0.5, 0.5), 
         inner_radius: float = 0.2, outer_radius: float = 0.4) -> float:
    cx, cy = center
    du, dv = u - cx, v - cy
    distance = math.sqrt(du * du + dv * dv)
    return 1.0 if inner_radius <= distance <= outer_radius else 0.0


def spiral(u: float, v: float, center: Tuple[float, float] = (0.5, 0.5), 
          turns: float = 4.0, thickness: float = 0.1) -> float:
    cx, cy = center
    du, dv = u - cx, v - cy
    
    distance = math.sqrt(du * du + dv * dv)
    if distance < 1e-6:
        return 1.0
    
    angle = math.atan2(dv, du)
    spiral_distance = distance * turns * 2.0 * math.pi - angle
    spiral_position = (spiral_distance % (2.0 * math.pi)) / (2.0 * math.pi)
    
    return 1.0 if spiral_position < thickness else 0.0


def stripes_horizontal(u: float, v: float, count: int = 4) -> float:
    return 1.0 if int(v * count) % 2 == 0 else 0.0


def stripes_vertical(u: float, v: float, count: int = 4) -> float:
    return 1.0 if int(u * count) % 2 == 0 else 0.0


def turbulence_noise(u: float, v: float, octaves: int = 4, frequency: float = 4.0) -> float:
    value = 0.0
    freq = frequency
    amp = 1.0
    
    for _ in range(octaves):
        value += abs(perlin_noise(u, v, freq) - 0.5) * amp
        freq *= 2.0
        amp *= 0.5
    
    return min(1.0, value)


def voronoi_noise(u: float, v: float, frequency: float = 8.0) -> float:
    x = u * frequency
    y = v * frequency
    
    cell_x = int(math.floor(x))
    cell_y = int(math.floor(y))
    
    min_dist = float('inf')
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            neighbor_x = cell_x + i
            neighbor_y = cell_y + j
            
            point_x = neighbor_x + hash_2d(neighbor_x, neighbor_y) / 4294967295.0
            point_y = neighbor_y + hash_2d(neighbor_y, neighbor_x) / 4294967295.0
            
            dist = math.sqrt((x - point_x) ** 2 + (y - point_y) ** 2)
            min_dist = min(min_dist, dist)
    
    return min(1.0, min_dist)