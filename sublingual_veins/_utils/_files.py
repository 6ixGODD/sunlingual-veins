from pathlib import Path
from typing import Union


def increment_path(path: Union[str, Path], separator: str = "-") -> Path:
    """
    Automatically increment path, i.e. weights/exp -> weights/exp{sep}2, weights/exp{sep}3, ...

    Args:
        path (str): path to be incremented
        separator (str): separator between path and number

    Returns:
        Path: incremented path
    """

    path = Path(path)
    if path.exists():
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 9999):
            p = f"{path}{separator}{n}{suffix}"
            if not Path(p).exists():
                path = Path(p)
                break
        path.mkdir(parents=True, exist_ok=True)  # make directory
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path
