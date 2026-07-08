"""Headless test harness for the DICOMator add-on.

The add-on's modules are written to run inside Blender, but most of the
numeric and DICOM-writing code only needs ``bpy``/``bmesh``/``mathutils`` to
*import*, not to run. This conftest installs minimal stub modules before any
project import and exposes :func:`load_module`, which loads add-on submodules
through a synthetic package so their relative imports resolve without
executing the package ``__init__`` (which requires a real ``bpy``).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_NAME = "dicomator_pkg"


class _Vector:
    """Tiny stand-in for ``mathutils.Vector`` (3 components)."""

    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = (float(v) for v in seq)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __len__(self):
        return 3


def _module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_stubs() -> None:
    if "mathutils" not in sys.modules:
        mathutils = _module("mathutils")
        mathutils.Vector = _Vector
        bvhtree = _module("mathutils.bvhtree")
        bvhtree.BVHTree = type("BVHTree", (), {})
        mathutils.bvhtree = bvhtree

    if "bpy" not in sys.modules:
        bpy = _module("bpy")
        bpy_types = _module("bpy.types")
        for name in (
            "Object",
            "Scene",
            "Context",
            "Depsgraph",
            "Operator",
            "Panel",
            "PropertyGroup",
            "Camera",
            "Mesh",
        ):
            setattr(bpy_types, name, type(name, (), {}))
        bpy.types = bpy_types

        def _prop_factory(**_kwargs):
            return None

        bpy_props = _module("bpy.props")
        for name in (
            "BoolProperty",
            "IntProperty",
            "FloatProperty",
            "StringProperty",
            "EnumProperty",
            "PointerProperty",
            "FloatVectorProperty",
            "CollectionProperty",
        ):
            setattr(bpy_props, name, _prop_factory)
        bpy.props = bpy_props

    if "bmesh" not in sys.modules:
        bmesh = _module("bmesh")
        bmesh_types = _module("bmesh.types")
        for name in ("BMesh", "BMEdge", "BMVert", "BMFace"):
            setattr(bmesh_types, name, type(name, (), {}))
        bmesh.types = bmesh_types
        bmesh.ops = _module("bmesh.ops")


_install_stubs()

if PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(PKG_NAME)
    _pkg.__path__ = [str(REPO_ROOT)]
    sys.modules[PKG_NAME] = _pkg


def load_module(name: str) -> types.ModuleType:
    """Load add-on submodule ``name`` (e.g. ``"artifacts"``) headlessly."""
    full = f"{PKG_NAME}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, REPO_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module
