import os
import winreg
from pathlib import Path
from typing import List

from global_constants import global_constants
from log import log
from utils import require_file, run_and_log_process


def start_toolbag() -> bool:
    require_file(global_constants.toolbag_dir / "toolbag.exe")
    run_and_log_process([str(global_constants.toolbag_dir / "toolbag.exe")])


def run(args: List[str] = []) -> int:
    return 0 if start_toolbag() else 1
