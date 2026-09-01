"""Exercise DICOMator registration and a small export in real Blender."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import bpy
import pydicom

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT.parent))

import DICOMator  # noqa: E402
from DICOMator import constants  # noqa: E402


def _assert_registration_cycle() -> None:
    """Verify registration is fully reversible within one Blender session."""

    DICOMator.register()
    try:
        assert hasattr(bpy.types.Scene, "dicomator_props")
        assert hasattr(bpy.types.Object, "dicomator_object_type")
        assert hasattr(bpy.types.Object, "dicomator_priority")
    finally:
        DICOMator.unregister()

    assert not hasattr(bpy.types.Scene, "dicomator_props")
    assert not hasattr(bpy.types.Object, "dicomator_object_type")


def _assert_failed_registration_rolls_back() -> None:
    """Force a late registration failure and verify that nothing leaks."""

    original_classes = DICOMator.classes
    DICOMator.classes = original_classes + (original_classes[0],)
    try:
        try:
            DICOMator.register()
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("duplicate class registration unexpectedly succeeded")
    finally:
        DICOMator.classes = original_classes

    assert not hasattr(bpy.types.Scene, "dicomator_props")
    assert not hasattr(bpy.types.Object, "dicomator_object_type")
    assert all(not getattr(cls, "is_registered", False) for cls in original_classes)


def _assert_cube_ct_export() -> None:
    """Voxelize a Blender cube, export CT slices, and read them back."""

    DICOMator.register()
    try:
        bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0.0, 0.0, 0.0))
        cube = bpy.context.active_object
        cube.name = "DICOMator Smoke Cube"
        cube.dicomator_hu = 250.0

        voxel_size = (0.01, 0.01, 0.01)
        volume, bbox_min, dimensions = DICOMator.voxelize_objects_to_hu(
            [cube],
            voxel_size=voxel_size,
            padding=1,
        )
        # Blender stores mesh coordinates as float32, so a nominal 100 mm
        # cube can land one representable value above 100 mm and require one
        # extra voxel after the ceiling operation. The grid must still be
        # cubic and include the two requested padding voxels.
        assert len(set(dimensions)) == 1, dimensions
        assert 12 <= dimensions[0] <= 13, dimensions
        assert volume.shape == dimensions
        assert int(volume.max()) == 250
        assert int(volume.min()) == constants.AIR_DENSITY

        with tempfile.TemporaryDirectory(prefix="dicomator-blender-smoke-") as output_dir:
            result = DICOMator.export_voxel_grid_to_dicom(
                volume,
                voxel_size,
                output_dir,
                bbox_min,
                direct_hu=True,
            )
            assert "error" not in result, result

            slices = sorted(Path(output_dir).glob("*.dcm"))
            assert len(slices) == dimensions[2]
            first = pydicom.dcmread(slices[0])
            assert first.Modality == "CT"
            assert (first.Rows, first.Columns) == dimensions[:2]
            assert first.pixel_array.shape == dimensions[:2]
    finally:
        DICOMator.unregister()


def main() -> None:
    assert constants.ensure_pydicom_available(), constants.get_pydicom_error()
    _assert_failed_registration_rolls_back()
    _assert_registration_cycle()
    _assert_cube_ct_export()
    print(f"DICOMator Blender smoke test passed in Blender {bpy.app.version_string}")


if __name__ == "__main__":
    main()
