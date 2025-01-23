from __future__ import annotations

from pathlib import Path


class GlobalConstants:
    def __del__(self):
        self.cleanup()

    def __init__(self):
        self.__lib_path = None
        self.__tools_path = None
        self.__root_path = None
        self.__plugins_path = None
        self.__toolbag_dir = None

    def cleanup(self):
        self.__lib_path = None
        self.__tools_path = None
        self.__root_path = None
        self.__plugins_path = None
        self.__toolbag_dir = None

    @property
    def lib_path(self) -> Path:
        if not isinstance(self.__lib_path, Path):
            self.__lib_path = self.tools_path / "Python" / "lib"
        return self.__lib_path

    @property
    def tools_path(self) -> Path:
        if not isinstance(self.__tools_path, Path):
            self.__tools_path = self.root_path / "Tools"
        return self.__tools_path

    @property
    def root_path(self) -> Path:
        file_path = Path(__file__).resolve()
        self.__root_path = file_path.parents[3]
        return self.__root_path

    @property
    def plugins_path(self) -> Path:
        if not isinstance(self.__plugins_path, Path):
            self.__plugins_path = self.root_path / "plugin"
        return self.__plugins_path

    @property
    def toolbag_dir(self) -> Path:
        if not isinstance(self.__toolbag_dir, Path):
            self.__toolbag_dir = Path("C:/Program Files/Marmoset/Toolbag 5")
        return self.__toolbag_dir


global_constants = GlobalConstants()
