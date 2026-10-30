import os
from pathlib import Path
from typing import Union


def clusterify_path(path: Union[str, Path], strict: bool = False) -> str:
    return str(path)


def is_github_ci() -> bool:
    return bool(os.environ.get("MOCK_CI", 0))
