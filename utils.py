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


def resolve_output_directory(output_dir: str) -> str:
    """Resolve Blender-relative paths into an absolute export directory."""

    resolved_dir = str(output_dir or "").strip()
    if not resolved_dir:
        return ""

    # bpy.path.abspath resolves the Blender-specific '//' prefix relative to
    # the open .blend file; unsaved files resolve relative to the current
    # working directory via os.path.abspath below.
    resolved_dir = bpy.path.abspath(resolved_dir)

    return os.path.abspath(os.path.normpath(resolved_dir))


__all__ = [
    "get_float_prop",
    "get_str_prop",
    "resolve_output_directory",
]
