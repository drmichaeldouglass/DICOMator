"""Blender entry point for the DICOMator add-on."""
from __future__ import annotations

import bpy
from bpy.props import FloatProperty, PointerProperty

from .artifacts import (
    add_gaussian_noise,
    add_metal_artifacts,
    add_motion_artifact,
    add_poisson_noise,
    add_ring_artifacts,
    apply_partial_volume_effect,
)
from .constants import MAX_HU_VALUE, MIN_HU_VALUE
from .dicom_export import export_voxel_grid_to_dicom
from .operators import MESH_OT_export_dicom
from .panels import (
    VIEW3D_PT_dicomator_export_settings,
    VIEW3D_PT_dicomator_orientation,
    VIEW3D_PT_dicomator_panel,
    VIEW3D_PT_dicomator_patient_info,
    VIEW3D_PT_dicomator_per_object_hu,
    VIEW3D_PT_dicomator_selection_info,
)
from .properties import DICOMatorProperties
from .voxelization import voxelize_mesh, voxelize_objects_to_hu

bl_info = {
    "name": "DICOMator",
    "author": "Michael Douglass",
    "version": (3, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > DICOMator",
    "description": "Converts mesh objects into DICOM CT files",
    "warning": "",
    "doc_url": "https://github.com/drmichaeldouglass/DICOMator",
    "category": "3D View",
}

classes = (
    DICOMatorProperties,
    MESH_OT_export_dicom,
    VIEW3D_PT_dicomator_panel,
    VIEW3D_PT_dicomator_selection_info,
    VIEW3D_PT_dicomator_per_object_hu,
    VIEW3D_PT_dicomator_patient_info,
    VIEW3D_PT_dicomator_orientation,
    VIEW3D_PT_dicomator_export_settings,
)


def register() -> None:  # pragma: no cover - Blender registration
    for cls in classes:
        bpy.utils.register_class(cls)

    if not hasattr(bpy.types.Scene, "dicomator_props"):
        bpy.types.Scene.dicomator_props = PointerProperty(type=DICOMatorProperties)

    if not hasattr(bpy.types.Object, "dicomator_hu"):
        bpy.types.Object.dicomator_hu = FloatProperty(
            name="HU",
            description="Assigned Hounsfield Units for this mesh",
            default=0.0,
            min=MIN_HU_VALUE,
            max=MAX_HU_VALUE,
            step=10,
            precision=0,
        )


def unregister() -> None:  # pragma: no cover - Blender registration
    if hasattr(bpy.types.Scene, "dicomator_props"):
        del bpy.types.Scene.dicomator_props

    if hasattr(bpy.types.Object, "dicomator_hu"):
        del bpy.types.Object.dicomator_hu

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


__all__ = [
    "add_gaussian_noise",
    "add_metal_artifacts",
    "add_motion_artifact",
    "add_poisson_noise",
    "add_ring_artifacts",
    "apply_partial_volume_effect",
    "export_voxel_grid_to_dicom",
    "voxelize_mesh",
    "voxelize_objects_to_hu",
    "register",
    "unregister",
]
