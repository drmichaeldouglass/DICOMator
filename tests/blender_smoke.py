"""Register and unregister DICOMator inside a real Blender process."""
from __future__ import annotations

import sys
from pathlib import Path

import bpy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT.parent))

import DICOMator  # noqa: E402
from DICOMator import constants  # noqa: E402


def main() -> None:
    assert constants.ensure_pydicom_available(), constants.get_pydicom_error()
    DICOMator.register()
    try:
        assert hasattr(bpy.types.Scene, "dicomator_props")
        assert hasattr(bpy.types.Object, "dicomator_object_type")
        assert hasattr(bpy.types.Object, "dicomator_priority")
    finally:
        DICOMator.unregister()
    print(f"DICOMator Blender smoke test passed in Blender {bpy.app.version_string}")


if __name__ == "__main__":
    main()
