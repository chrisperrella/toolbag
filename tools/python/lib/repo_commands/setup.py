import argparse
import os
from pathlib import Path
from typing import List

from global_constants import global_constants
from log import log
from utils import make_junction, strip_suffix


def setup() -> bool:
    local_appdata = Path.home() / "AppData" / "Local"
    plugins_path = local_appdata / "Marmoset Toolbag 5" / "plugins"
    if not plugins_path.exists():
        log.error(f'Path "{plugins_path}" does not exist!')
        return False
    for plugin in global_constants.plugins_path.iterdir():
        if plugin.is_dir():
            junction_path = plugins_path / plugin.name
            if junction_path.exists():
                log.warning(f'Path "{junction_path}" already exists!')
                continue
            if not make_junction(plugin, junction_path):
                return False
    return True


def run(args: List[str] = []) -> int:
    parser = argparse.ArgumentParser(
        prog=strip_suffix(os.path.basename(__file__)),
    )
    parser.parse_args(args=args)
    return 0 if setup() else 1
