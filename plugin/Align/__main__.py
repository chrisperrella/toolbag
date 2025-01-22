import os
from typing import List, Optional, Union

import mset


class AlignPlugin:
    def __init__(self) -> None:
        self.window: mset.UIWindow = mset.UIWindow("Align Tools")
        self.window.width = 175
        self._render()
        self.window.visible = True

    @staticmethod
    def _get_selected_objects(
        multi_selection_required: bool = True,
    ) -> List[mset.MeshObject]:
        selected_objects = mset.getSelectedObjects()
        selected_mesh_objects = [
            obj for obj in selected_objects if isinstance(obj, mset.MeshObject)
        ]
        if multi_selection_required and len(selected_mesh_objects) < 2:
            mset.err("[AlignPlugin]: At least two mesh objects must be selected.")
        if not selected_mesh_objects:
            mset.err("[AlignPlugin]: No MeshObject selected.")
        return selected_mesh_objects

    @staticmethod
    def _axis_to_index(axis: Union[str, int]) -> int:
        axis_map = {"x": 0, "y": 1, "z": 2}
        return axis_map.get(axis, axis)

    def _align_to_value(self, axis: int, value: float) -> None:
        selected_objects = self._get_selected_objects()
        for prim in selected_objects:
            new_position = prim.position.copy()
            new_position[axis] = value
            prim.position = new_position
            mset.log(f"[AlignPlugin]: Aligning {prim.name} to {new_position}")

    def align(self, axis: int = 0) -> None:
        selected_objects = self._get_selected_objects()
        target_prim_axis_value: float = selected_objects[0].position[axis]
        self._align_to_value(axis, target_prim_axis_value)

    def align_to_min(self, axis: int = 0) -> None:
        selected_objects = self._get_selected_objects()
        min_value: float = min(prim.position[axis] for prim in selected_objects)
        self._align_to_value(axis, min_value)

    def align_to_max(self, axis: int = 0) -> None:
        selected_objects = self._get_selected_objects()
        max_value: float = max(prim.position[axis] for prim in selected_objects)
        self._align_to_value(axis, max_value)

    def distribute(self, axis: int = 0, spacing: float = 0.1) -> None:
        selected_objects = self._get_selected_objects()
        target_prim_axis_value: float = selected_objects[0].position[axis]
        for idx, prim in enumerate(selected_objects[1:]):
            new_position = prim.position.copy()
            new_position[axis] = target_prim_axis_value + (idx + 1) * spacing
            prim.position = new_position

    def _render(self) -> None:
        self.tool_logo_button: mset.UIButton = mset.UIButton()
        self.tool_logo_button.frameless = True
        self.tool_logo_button.lit = False

        self.axis_align_label: mset.UILabel = mset.UILabel(
            "Align to First Selection Axis:"
        )
        self.axis_align_x: mset.UIButton = mset.UIButton()
        self.axis_align_x.text = "X"
        self.axis_align_x.onClick = lambda: self.align(axis=0)

        self.axis_align_y: mset.UIButton = mset.UIButton()
        self.axis_align_y.text = "Y"
        self.axis_align_y.onClick = lambda: self.align(axis=1)

        self.axis_align_z: mset.UIButton = mset.UIButton()
        self.axis_align_z.text = "Z"
        self.axis_align_z.onClick = lambda: self.align(axis=2)

        self.axis_align_min_label: mset.UILabel = mset.UILabel("Align to Axis Min:")
        self.axis_align_min_x: mset.UIButton = mset.UIButton()
        self.axis_align_min_x.text = "X"
        self.axis_align_min_x.onClick = lambda: self.align_to_min(0)

        self.axis_align_min_y: mset.UIButton = mset.UIButton()
        self.axis_align_min_y.text = "Y"
        self.axis_align_min_y.onClick = lambda: self.align_to_min(1)

        self.axis_align_min_z: mset.UIButton = mset.UIButton()
        self.axis_align_min_z.text = "Z"
        self.axis_align_min_z.onClick = lambda: self.align_to_min(2)

        self.axis_align_max_label: mset.UILabel = mset.UILabel("Align to Axis Max:")
        self.axis_align_max_x: mset.UIButton = mset.UIButton()
        self.axis_align_max_x.text = "X"
        self.axis_align_max_x.onClick = lambda: self.align_to_max(0)

        self.axis_align_max_y: mset.UIButton = mset.UIButton()
        self.axis_align_max_y.text = "Y"
        self.axis_align_max_y.onClick = lambda: self.align_to_max(1)

        self.axis_align_max_z: mset.UIButton = mset.UIButton()
        self.axis_align_max_z.text = "Z"
        self.axis_align_max_z.onClick = lambda: self.align_to_max(2)

        self.axis_distribute_label: mset.UILabel = mset.UILabel(
            "Distribute Along Axis:"
        )
        self.axis_distribute_x: mset.UIButton = mset.UIButton()
        self.axis_distribute_x.text = "X"
        self.axis_distribute_x.onClick = lambda: self.distribute(
            axis=0, spacing=self.axis_distribute_spacing.value
        )

        self.axis_distribute_y: mset.UIButton = mset.UIButton()
        self.axis_distribute_y.text = "Y"
        self.axis_distribute_y.onClick = lambda: self.distribute(
            axis=1, spacing=self.axis_distribute_spacing.value
        )

        self.axis_distribute_z: mset.UIButton = mset.UIButton()
        self.axis_distribute_z.text = "Z"
        self.axis_distribute_z.onClick = lambda: self.distribute(
            axis=2, spacing=self.axis_distribute_spacing.value
        )

        self.axis_distribute_spacing: mset.UISliderFloat = mset.UISliderFloat(
            min=0, max=250, name="Spacing"
        )

        self.window.addElement(self.axis_align_label)
        self.window.addReturn()
        self.window.addElement(self.axis_align_x)
        self.window.addElement(self.axis_align_y)
        self.window.addElement(self.axis_align_z)
        self.window.addReturn()
        self.window.addReturn()
        self.window.addElement(self.axis_align_min_label)
        self.window.addReturn()
        self.window.addElement(self.axis_align_min_x)
        self.window.addElement(self.axis_align_min_y)
        self.window.addElement(self.axis_align_min_z)
        self.window.addReturn()
        self.window.addReturn()
        self.window.addElement(self.axis_align_max_label)
        self.window.addReturn()
        self.window.addElement(self.axis_align_max_x)
        self.window.addElement(self.axis_align_max_y)
        self.window.addElement(self.axis_align_max_z)
        self.window.addReturn()
        self.window.addElement(self.axis_distribute_label)
        self.window.addReturn()
        self.window.addElement(self.axis_distribute_x)
        self.window.addElement(self.axis_distribute_y)
        self.window.addElement(self.axis_distribute_z)
        self.window.addReturn()
        self.window.addReturn()
        self.window.addElement(self.axis_distribute_spacing)

    def _get_icon_from_gui_folder(self, icon_name: str) -> Optional[str]:
        path: str = os.path.join(os.getcwd(), "data", "gui", "control", icon_name)
        return path if os.path.exists(path) else None

    def _get_icon_from_plugin_folder(self, icon_name: str) -> str:
        return os.path.join(os.getcwd(), "data", "plugin", "Align", "gui", icon_name)

    def _shutdown(self) -> None:
        mset.shutdownPlugin()


if __name__ == "__main__":
    AlignPlugin()
