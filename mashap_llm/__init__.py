from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "mas",
    "envs",
    "eval",
    "critics",
    "runner",
    "scripts",
    "algorithms",
    "utils",
    "config",
]

def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
