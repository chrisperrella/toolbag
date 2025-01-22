import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

import toml


def are_lists_same(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for index in range(len(a)):
        field_a = a[index]
        field_b = b[index]
        if isinstance(field_a, list) and isinstance(field_b, list):
            if not are_lists_same(field_a, field_b):
                return False
        if isinstance(field_a, dict) and isinstance(field_b, dict):
            if not are_objects_same(field_a, field_b):
                return False
        if type(field_a) is not type(field_b):
            return False
        if field_a != field_b:
            return False
    return True


def are_objects_same(a: dict, b: dict) -> bool:
    keys = set().union(set(a.keys())).union(set(b.keys()))
    for key in keys:
        if key not in a or key not in b:
            return False
        field_a = a[key]
        field_b = b[key]
        if isinstance(field_a, list) and isinstance(field_b, list):
            if not are_lists_same(field_a, field_b):
                return False
        if isinstance(field_a, dict) and isinstance(field_b, dict):
            if not are_objects_same(field_a, field_b):
                return False
        if type(field_a) is not type(field_b):
            return False
        if field_a != field_b:
            return False
    return True


def copy_directory(src: Union[Path, str], dst: Union[Path, str]) -> bool:
    from .log import log

    try:
        shutil.copytree(src, dst)
    except Exception as e:
        log.error(f'Failed to copy directory "{src}" to "{dst}"!')
        log.error(str(e))
        return False
    return True


def find_folders_with_name(
    results: List[Path],
    path: Union[Path, str],
    folder_name: str,
    recursive: bool = False,
) -> List[Path]:
    from .log import log

    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        return True
    if not path.is_dir() or is_junction(path):
        log.error(f'Path "{path}" is not a directory nor a junction!')
        return False
    folder_name_lower = folder_name.lower()

    def _find_folders_with_name(r: List[Path], p: Path):
        for entry in os.listdir(p):
            child_path = p / entry
            child_path.is_symlink()
            if child_path.is_dir() or is_junction(child_path):
                if entry.lower() == folder_name_lower:
                    r.append(child_path)
                elif recursive:
                    if not _find_folders_with_name(r, child_path):
                        return False
        return True

    return _find_folders_with_name(results, path)


def get_suffix(filename: Union[Path, str], separator: str = ".") -> str:
    if not isinstance(filename, str):
        filename = str(filename)
    index = filename.rfind(separator)
    next_index = index + 1
    return filename[next_index:] if index >= 0 else ""


def is_junction(path: Union[Path, str]) -> bool:
    try:
        return True if os.readlink(str(path)) else False
    except OSError:
        return False


def load_merged_toml_files(*args: List[Union[Path, str]]) -> dict:
    from .log import log

    toml_obj = {}
    for path in args:
        if not isinstance(path, Path):
            path = Path(path)
        if path.is_file():
            with open(path, "r") as file:
                try:
                    obj = toml.loads(file.read())
                    merge_objects(toml_obj, obj)
                except TypeError as e:
                    log.warning(f'Failed to parse toml file "{path}"! ({e})')
                    obj = {}
                except toml.TomlDecodeError as e:
                    log.warning(f'Failed to parse toml file "{path}"! ({e})')
                    obj = {}
    return toml_obj


def merge_objects(dst: dict, src: dict):
    for key in src.keys():
        if key not in dst:
            dst[key] = src[key]
        elif isinstance(dst[key], dict) and isinstance(src[key], dict):
            merge_objects(dst[key], src[key])
        else:
            dst[key] = src[key]


def remove_path(path: Union[Path, str], remove_readonly_files: bool = False) -> bool:
    from .log import log

    def error_fn(fn: callable, path: str, exc_info: tuple):
        if remove_readonly_files:
            os.chmod(path, stat.S_IWRITE)
            fn(path)
        else:
            log.error(str(exc_info[1]))

    if not isinstance(path, Path):
        path = Path(path)
    if is_junction(path):
        path.unlink()
    elif path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, onerror=error_fn)
    elif path.exists():
        if remove_readonly_files:
            os.chmod(path, stat.S_IWRITE)
        path.unlink()
    if os.path.exists(path):
        log.error(f'Failed to remove "{path}"!')
        return False
    return True


def require_file(path: Union[Path, str]):
    if not isinstance(path, Path):
        path = Path(path)
    if not str(path).strip():
        raise ValueError("No path specified!")
    if not path.exists():
        raise FileNotFoundError(f'Path "{path}" does not exist!')
    if not path.is_file():
        raise IsADirectoryError(f'Path "{path}" is not a file!')


def run_and_log_process(
    command: list,
    is_warning_fn: Callable[[str], bool] = lambda _: False,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    from .log import log

    def do_log(p: subprocess.Popen):
        stdout = p.stdout.readline()
        while stdout:
            msg = stdout.rstrip()
            log_fn = log.warning if is_warning_fn(msg) else log.info
            log_fn(msg)
            stdout = p.stdout.readline()
        stderr = p.stderr.readline()
        if stderr:
            log_fn = log.error
            if is_warning_fn(stderr):
                log_fn = log.warning
            while stderr:
                log_fn(stderr.rstrip())
                stderr = p.stderr.readline()

    command_str = " ".join(command)
    log.debug(command_str)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        env=env,
    )
    while process.poll() is None:
        do_log(process)
    do_log(process)
    if process.returncode != 0:
        log.error(f'Failed to run "{command_str}" ({process.returncode})!')
    return process


def strip_suffix(filename: str, separator: str = ".") -> str:
    index = filename.rfind(separator)
    return filename[:index] if index >= 0 else filename
