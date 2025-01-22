import argparse
import os
from typing import List

from global_constants import global_constants
from log import log
from utils import strip_suffix

def setup() -> bool:
    return True

def run(args: List[str] = []) -> int:
    parser = argparse.ArgumentParser(
        prog=strip_suffix(os.path.basename(__file__)),
    )
    parser.parse_args(args=args)
    return 0 if setup() else 1
