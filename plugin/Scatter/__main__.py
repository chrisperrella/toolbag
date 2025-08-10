import mset
from scatter_core import scatter_with_primitive, scatter_with_selected_prototype
from scatter_tests import run_tests

NUM_POINTS = 1000
SEED = None

class ScatterUI:
    def __init__(self) -> None:
        self.window = mset.UIWindow("Scatter Tools")
        self.btn_scatter_sphere = mset.UIButton("Scatter: Sphere")
        self.btn_scatter_proto = mset.UIButton("Scatter: Prototype")
        self.btn_run_tests = mset.UIButton("Run Tests")
        self.btn_scatter_sphere.onClick = self._on_scatter_sphere
        self.btn_scatter_proto.onClick = self._on_scatter_proto
        self.btn_run_tests.onClick = self._on_run_tests
        self.window.addElement(self.btn_scatter_sphere)
        self.window.addElement(self.btn_scatter_proto)
        self.window.addElement(self.btn_run_tests)

    def _on_scatter_proto(self):
        scatter_with_selected_prototype(num_points=NUM_POINTS, seed=SEED)

    def _on_scatter_sphere(self):
        scatter_with_primitive(num_points=NUM_POINTS, seed=SEED)

    def _on_run_tests(self):
        run_tests(num_points=NUM_POINTS if NUM_POINTS > 100 else 500)


if __name__ == "__main__":
    ScatterUI()
