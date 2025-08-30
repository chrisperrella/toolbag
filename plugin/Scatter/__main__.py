import random

import mset
from scatter_core import create_scene_and_scatter


class ScatterUI:
    def __init__(self) -> None:
        self._num_points = 1000
        self._seed = None
        self._create_ui()

    def _create_ui(self) -> None:
        self.window = mset.UIWindow("Scatter System")
        self.window.width = 320
        
        self._create_surface_generation_section()
        self._create_scatter_parameters_section()
        self._create_action_section()

    def _create_surface_generation_section(self) -> None:
        surface_drawer = mset.UIDrawer(name="Surface Generation")
        surface_window = mset.UIWindow(name="")
        surface_drawer.containedControl = surface_window
        
        surface_type_label = mset.UILabel("Surface Type:")
        surface_window.addElement(surface_type_label)
        surface_window.addReturn()
        
        self.surface_dropdown = mset.UIListBox("Surface Type")
        self.surface_dropdown.addItem("Procedural Plane")
        self.surface_dropdown.addItem("UV Sphere")
        self.surface_dropdown.addItem("Terrain (Noise)")
        self.surface_dropdown.selectItemByName("Procedural Plane")
        surface_window.addElement(self.surface_dropdown)
        
        self.window.addElement(surface_drawer)
        self.window.addReturn()

    def _create_scatter_parameters_section(self) -> None:
        scatter_drawer = mset.UIDrawer(name="Scatter Parameters")
        scatter_window = mset.UIWindow(name="")
        scatter_drawer.containedControl = scatter_window
        
        points_label = mset.UILabel("Point Count:")
        scatter_window.addElement(points_label)
        scatter_window.addReturn()
        
        self.points_slider = mset.UISliderInt()
        self.points_slider.min = 10
        self.points_slider.max = 5000
        self.points_slider.value = self._num_points
        self.points_slider.onChange = self._on_points_changed
        scatter_window.addElement(self.points_slider)
        scatter_window.addReturn()
        scatter_window.addReturn()
        
        seed_label = mset.UILabel("Random Seed:")
        scatter_window.addElement(seed_label)
        scatter_window.addReturn()
        
        seed_help_label = mset.UILabel("(0 for random generation)")
        scatter_window.addElement(seed_help_label)
        scatter_window.addReturn()
        
        self.seed_field = mset.UITextFieldInt()
        self.seed_field.value = 0
        self.seed_field.onChange = self._on_seed_changed
        scatter_window.addElement(self.seed_field)
        
        self.btn_random_seed = mset.UIButton("Random")
        self.btn_random_seed.onClick = self._on_generate_random_seed
        scatter_window.addElement(self.btn_random_seed)
        
        self.window.addElement(scatter_drawer)
        self.window.addReturn()

    def _create_action_section(self) -> None:
        self.btn_generate = mset.UIButton("Generate Scatter")
        self.btn_generate.onClick = self._on_execute
        self.window.addElement(self.btn_generate)
        self.window.addReturn()
        
        self.btn_close = mset.UIButton("Close")
        self.btn_close.onClick = lambda: mset.shutdownPlugin()
        self.window.addElement(self.btn_close)

    def _on_points_changed(self):
        self._num_points = self.points_slider.value

    def _on_seed_changed(self):
        self._seed = self.seed_field.value if self.seed_field.value != 0 else None

    def _on_generate_random_seed(self):
        random_seed = random.randint(1, 999999)
        self.seed_field.value = random_seed
        self._seed = random_seed

    def _on_execute(self):
        surface_mapping = {
            0: "plane",
            1: "sphere", 
            2: "terrain"
        }
        
        selected_index = self.surface_dropdown.selectedItem
        selected_surface = surface_mapping.get(selected_index, "plane")
        
        create_scene_and_scatter(selected_surface, self._num_points, self._seed)


if __name__ == "__main__":
    ScatterUI()