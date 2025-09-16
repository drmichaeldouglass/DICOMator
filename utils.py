"""Utility helpers for Blender property access and UI refresh."""
from __future__ import annotations

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


__all__ = ["get_float_prop", "get_str_prop", "force_ui_redraw"]
