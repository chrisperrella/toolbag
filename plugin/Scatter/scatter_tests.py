from pathlib import Path
from typing import List, Tuple
import mset

from scatter_core import (
    ScatterSurface,
    ScatterPoint,
    ScatterMask,
    ScatterTriangle,
    build_uvsphere_instance,
)

def _checkerboard(u: float, v: float, squares: int = 2) -> float:
    return 1.0 if (int(u * squares) + int(v * squares)) % 2 == 0 else 0.0

def _dark_ring(u: float, v: float) -> float:
    cx, cy = 0.5, 0.5
    du, dv = u - cx, v - cy
    r = (du * du + dv * dv) ** 0.5
    return 0.0 if 0.35 < r < 0.45 else 1.0

def _run_once(num_points: int, seed: int) -> Tuple[int, List[List[float]]]:
    selected = mset.getSelectedObjects()
    if not selected:
        mset.log("No objects selected.")
        return 0, []
    surface_obj = selected[0]
    surface = ScatterSurface(surface_obj, seed=seed)
    surface.add_scatter_mask(_checkerboard)
    surface.add_scatter_mask(_dark_ring, blend_method="multiply")

    from debug_tools import (
        save_callable_mask_as_image,
        save_combined_mask_image,
        initialize_csv_logger,
        log_csv_sample,
        save_uv_scatter_image,
    )

    save_callable_mask_as_image(_checkerboard, Path("C:/Temp/mask_checkerboard.png"))
    save_combined_mask_image(surface, Path("C:/Temp/mask_combined.png"))

    csv_path = Path("C:/Temp/scatter_harness.csv")
    initialize_csv_logger(csv_path)

    instance = build_uvsphere_instance()

    accepted = 0
    for _ in range(num_points):
        point = surface.generate_scatter_point(mesh=instance)
        if point is not None:
            accepted += 1
            log_csv_sample(csv_path, point.uv[0], point.uv[1], 1.0, True, 0)

    surface.duplicate_mesh_objects_to_points()

    save_uv_scatter_image(surface, Path("C:/Temp/uv_scatter.png"), size=1024, point_radius=2)

    uvs = [p.uv for p in surface.scatter_points]
    return accepted, uvs

def run_tests(num_points: int = 500) -> None:
    mset.log("Running scatter tests.")
    a_count, a_uvs = _run_once(num_points=num_points, seed=12345)
    b_count, b_uvs = _run_once(num_points=num_points, seed=12345)
    if a_count != b_count:
        mset.log(f"Determinism failed: counts differ. {a_count} vs {b_count}")
        return
    same_uvs = all(
        abs(ua - ub) <= 1e-7
        for pair in zip(a_uvs, b_uvs)
        for ua, ub in zip(pair[0], pair[1])
    )
    if not same_uvs:
        mset.log("Determinism failed: UVs differ with same seed.")
        return
    c_count, _ = _run_once(num_points=num_points, seed=67890)
    if c_count == a_count:
        mset.log("Variability warning: counts identical across different seed.")
    else:
        mset.log("Determinism passed and variability check succeeded.")
