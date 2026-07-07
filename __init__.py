"""Blender entry point for the DICOMator add-on."""
from __future__ import annotations

import importlib
import sys

import bpy
from bpy.props import EnumProperty, FloatProperty, PointerProperty

# Blender keeps add-on submodules alive in sys.modules across disable/enable
# cycles. Reload any that are already present (a no-op on first load) before
# the named imports below so newly added panel/operator classes are visible
# without restarting Blender.
for _module_name in (
    "constants",
    "utils",
    "artifacts",
    "dicom_export",
    "drr",
    "rtdose_export",
    "rtstruct_export",
    "voxelization",
    "properties",
    "operators",
    "panels",
):
    _qualified_name = f"{__name__}.{_module_name}"
    if _qualified_name in sys.modules:
        importlib.reload(sys.modules[_qualified_name])

from .artifacts import (
    add_gaussian_noise,
    add_metal_artifacts,
    add_motion_artifact,
    add_poisson_noise,
    add_ring_artifacts,
    apply_partial_volume_effect,
)
from .constants import DICOM_OBJECT_TYPE_ITEMS, MATERIAL_ITEMS, MAX_HU_VALUE, MIN_HU_VALUE, ROI_TYPE_ITEMS
from .dicom_export import export_projection_to_dicom, export_voxel_grid_to_dicom
from .drr import generate_drr_from_hu_volume, resolve_drr_detector_size
from .operators import MESH_OT_export_dicom
from .rtdose_export import export_rtdose_to_dicom
from .rtstruct_export import export_rtstruct_to_dicom
from .panels import (
    VIEW3D_PT_dicomator_artifacts,
    VIEW3D_PT_dicomator_export_settings,
    VIEW3D_PT_dicomator_panel,
    VIEW3D_PT_dicomator_patient_info,
    VIEW3D_PT_dicomator_per_object_hu,
    VIEW3D_PT_dicomator_selection_info,
)
from .properties import DICOMatorProperties, update_object_material
from .voxelization import voxelize_mesh, voxelize_objects_to_dose, voxelize_objects_to_hu

# Legacy add-on metadata; ignored when installed as an extension (the
# blender_manifest.toml is authoritative). Kept in sync per AGENTS.md.
bl_info = {
    "name": "DICOMator",
    "author": "Michael Douglass",
    "version": (3, 3, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > DICOMator",
    "description": "Converts mesh objects into synthetic CT/MR series or camera-based DRR DICOM images",
    "warning": "",
    "doc_url": "https://github.com/drmichaeldouglass/DICOMator",
    "category": "3D View",
}

classes = (
    DICOMatorProperties,
    MESH_OT_export_dicom,
    VIEW3D_PT_dicomator_panel,
    VIEW3D_PT_dicomator_per_object_hu,
    VIEW3D_PT_dicomator_export_settings,
    VIEW3D_PT_dicomator_artifacts,
    VIEW3D_PT_dicomator_patient_info,
    VIEW3D_PT_dicomator_selection_info,
)


def register() -> None:  # pragma: no cover - Blender registration
    for cls in classes:
        bpy.utils.register_class(cls)

    # Assign unconditionally: guarding with hasattr() would keep a stale
    # definition from a previous registration (e.g. an older add-on version)
    # bound instead of the current one. Re-assignment is idempotent.
    bpy.types.Scene.dicomator_props = PointerProperty(type=DICOMatorProperties)

    bpy.types.Object.dicomator_hu = FloatProperty(
        name="HU",
        description="Assigned Hounsfield Units for this mesh",
        default=0.0,
        min=MIN_HU_VALUE,
        max=MAX_HU_VALUE,
        step=10,
        precision=0,
    )

    bpy.types.Object.dicomator_material = EnumProperty(
        name="Material",
        description="Select a predefined tissue/material",
        items=MATERIAL_ITEMS,
        default="CUSTOM",
        update=update_object_material,
    )

    bpy.types.Object.dicomator_object_type = EnumProperty(
        name="DICOM Object Type",
        description="Specifies whether this mesh contributes to image, RT Dose, or RT Structure exports",
        items=DICOM_OBJECT_TYPE_ITEMS,
        default="CT",
    )

    bpy.types.Object.dicomator_dose = FloatProperty(
        name="Dose (Gy)",
        description="Absorbed dose assigned to voxels within this mesh when exported as RT Dose",
        default=0.0,
        min=0.0,
        soft_max=80.0,
        max=200.0,
        step=10,
        precision=2,
    )

    bpy.types.Object.dicomator_roi_type = EnumProperty(
        name="ROI Type",
        description="DICOM RT ROI interpreted type (RTROIInterpretedType) for this structure",
        items=ROI_TYPE_ITEMS,
        default="OAR",
    )


def unregister() -> None:  # pragma: no cover - Blender registration
    if hasattr(bpy.types.Scene, "dicomator_props"):
        del bpy.types.Scene.dicomator_props

    if hasattr(bpy.types.Object, "dicomator_roi_type"):
        del bpy.types.Object.dicomator_roi_type

    if hasattr(bpy.types.Object, "dicomator_dose"):
        del bpy.types.Object.dicomator_dose

    if hasattr(bpy.types.Object, "dicomator_object_type"):
        del bpy.types.Object.dicomator_object_type

    if hasattr(bpy.types.Object, "dicomator_material"):
        del bpy.types.Object.dicomator_material

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
    "export_projection_to_dicom",
    "export_rtdose_to_dicom",
    "export_rtstruct_to_dicom",
    "export_voxel_grid_to_dicom",
    "generate_drr_from_hu_volume",
    "voxelize_mesh",
    "voxelize_objects_to_dose",
    "voxelize_objects_to_hu",
    "resolve_drr_detector_size",
    "register",
    "unregister",
]
