from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parent


def _candidate_runtime_dirs() -> list[Path]:
    if not is_frozen():
        return [Path(__file__).resolve().parent]

    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        exe_dir,
        exe_dir.parent,
        exe_dir.parent.parent,
        bundle_dir(),
        bundle_dir().parent,
        bundle_dir().parent.parent,
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def runtime_dir() -> Path:
    markers = (".env", "models", "voice_samples")
    for candidate in _candidate_runtime_dirs():
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return _candidate_runtime_dirs()[0]


def bundle_path(*parts: str) -> Path:
    return bundle_dir().joinpath(*parts)


def runtime_path(*parts: str) -> Path:
    return runtime_dir().joinpath(*parts)
