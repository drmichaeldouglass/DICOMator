"""Utility helpers for Blender property access and path resolution."""
from __future__ import annotations

import os

import bpy


def get_float_prop(props, name: str, default: float) -> float:
    """Safely read a float property, returning ``default`` on failure."""
    try:
        val = getattr(props, name)
        return float(val)
    except Exception:
        return float(default)


def get_str_prop(props, name: str, default: str) -> str:
    """Safely read a string property, returning ``default`` on failure."""
    try:
        val = getattr(props, name)
        return str(val) if val is not None else str(default)
    except Exception:
        return str(default)


def _blend_file_is_saved() -> bool:
    """Return True when the open .blend file has been saved to disk."""

    return bool(getattr(getattr(bpy, "data", None), "filepath", ""))


def resolve_output_directory(output_dir: str) -> str:
    """Resolve Blender-relative paths into an absolute export directory."""

    resolved_dir = str(output_dir or "").strip()
    if not resolved_dir:
        return ""

    # bpy.path.abspath resolves the Blender-specific '//' prefix relative to
    # the open .blend file. An unsaved file has no folder to be relative to,
    # and Blender would fall back to the process working directory: '/' when
    # launched from the macOS Finder, the install folder on Windows. Neither
    # is writable, so the default '//DICOM_Export' would fail on a new scene.
    # The home folder is used instead; the Export panel shows the result.
    if resolved_dir.startswith("//") and not _blend_file_is_saved():
        resolved_dir = os.path.join(os.path.expanduser("~"), resolved_dir[2:].lstrip("/\\"))
    else:
        resolved_dir = bpy.path.abspath(resolved_dir)
    # A typed "~/DICOM" means the home folder, not a directory named "~".
    resolved_dir = os.path.expanduser(resolved_dir)

    return os.path.abspath(os.path.normpath(resolved_dir))


__all__ = [
    "get_float_prop",
    "get_str_prop",
    "resolve_output_directory",
]
