"""Utility helpers for Blender property access and UI refresh."""
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

    if resolved_dir.startswith('//'):
        relative_path = resolved_dir[2:].replace('/', os.sep).replace('\\', os.sep)
        if bpy.data.filepath:
            blend_dir = os.path.dirname(bpy.data.filepath)
            resolved_dir = os.path.join(blend_dir, relative_path)
        else:
            resolved_dir = os.path.join(os.getcwd(), relative_path)

    return os.path.abspath(os.path.normpath(resolved_dir))


def force_ui_redraw() -> None:
    """Trigger a UI redraw so timeline/frame changes are visible to the user."""
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN', iterations=1)
    except Exception:
        window_manager = bpy.context.window_manager
        if window_manager:
            for window in window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()


__all__ = [
    "get_float_prop",
    "get_str_prop",
    "resolve_output_directory",
    "force_ui_redraw",
]
