"""UI panel definitions for the DICOMator add-on."""
from __future__ import annotations

import math

import bpy
from bpy.types import Context, Panel
from mathutils import Vector

from .constants import MRI_MODALITIES, ensure_pydicom_available, get_pydicom_error
from .drr import resolve_drr_detector_size
from .utils import get_float_prop, get_str_prop, resolve_output_directory


def _selected_meshes(context: Context) -> list[bpy.types.Object]:
    """Return selected mesh objects, falling back to the active mesh."""

    selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
    active_obj = context.active_object
    if not selected and active_obj and active_obj.type == 'MESH':
        selected = [active_obj]
    return selected


def _export_type_counts(objects: list[bpy.types.Object]) -> dict[str, int]:
    """Count selected meshes by their DICOM export role."""

    counts = {"CT": 0, "RTDOSE": 0, "RTSTRUCT": 0}
    for obj in objects:
        obj_type = getattr(obj, "dicomator_object_type", "CT")
        if obj_type in counts:
            counts[obj_type] += 1
    return counts


def _export_summary(objects: list[bpy.types.Object]) -> str:
    """Return a compact summary of selected export roles."""

    counts = _export_type_counts(objects)
    parts = []
    if counts["CT"]:
        parts.append(f"Image {counts['CT']}")
    if counts["RTDOSE"]:
        parts.append(f"Dose {counts['RTDOSE']}")
    if counts["RTSTRUCT"]:
        parts.append(f"Struct {counts['RTSTRUCT']}")
    return " | ".join(parts)


def _selection_bounds(objects: list[bpy.types.Object]) -> tuple[float, float, float]:
    """Return selected-object dimensions in metres."""

    bbox_corners = []
    for obj in objects:
        bbox_corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)
    return max_x - min_x, max_y - min_y, max_z - min_z


def _grid_estimate(
    dimensions_m: tuple[float, float, float],
    props: bpy.types.PropertyGroup,
) -> tuple[int, int, int, int] | None:
    """Estimate voxel dimensions and total voxel count for the selection."""

    obj_width, obj_height, obj_depth = dimensions_m
    lateral_mm = get_float_prop(props, "lateral_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
    axial_mm = get_float_prop(props, "axial_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
    if lateral_mm <= 0.0 or axial_mm <= 0.0:
        return None

    vx = lateral_mm * 0.001
    vy = lateral_mm * 0.001
    vz = axial_mm * 0.001
    est_width = int(math.ceil((obj_width + 2 * vx) / vx))
    est_height = int(math.ceil((obj_height + 2 * vy) / vy))
    est_depth = int(math.ceil((obj_depth + 2 * vz) / vz))
    total_voxels = est_width * est_height * est_depth
    return est_width, est_height, est_depth, total_voxels


def _draw_export_action(layout: bpy.types.UILayout, context: Context) -> None:
    """Draw the primary export action and blocking status."""

    props = context.scene.dicomator_props
    button_text = "Export DICOM"

    if not ensure_pydicom_available():
        layout.label(text="pydicom unavailable", icon='ERROR')
        error_detail = get_pydicom_error()
        if error_detail:
            layout.label(text=error_detail[:120], icon='INFO')
        return

    export_dir = resolve_output_directory(get_str_prop(props, "export_directory", ""))
    if not export_dir or not export_dir.strip():
        layout.label(text="Choose an export folder", icon='FILE_FOLDER')
        return

    if not (context.active_object and context.active_object.type == 'MESH'):
        layout.label(text="Select a mesh", icon='INFO')
        return

    requested = (
        bool(getattr(props, "export_image_series", True))
        or bool(getattr(props, "export_drr", False))
        or bool(getattr(props, "export_rtdose", False))
        or bool(getattr(props, "export_rtstruct", False))
    )
    if not requested:
        layout.label(text="Choose at least one output", icon='INFO')
        return

    selected_meshes = _selected_meshes(context)
    counts = _export_type_counts(selected_meshes)
    if (getattr(props, "export_image_series", True) or getattr(props, "export_drr", False)) and not counts["CT"]:
        layout.label(text="No image mesh selected", icon='ERROR')
        return
    if getattr(props, "export_rtdose", False) and not counts["RTDOSE"]:
        layout.label(text="No dose mesh selected", icon='ERROR')
        return
    if getattr(props, "export_rtstruct", False) and not counts["RTSTRUCT"]:
        layout.label(text="No structure mesh selected", icon='ERROR')
        return
    if getattr(props, "export_drr", False):
        camera_obj = context.scene.camera
        if camera_obj is None or camera_obj.type != 'CAMERA':
            layout.label(text="Set a scene camera", icon='ERROR')
            return

    estimate = _grid_estimate(_selection_bounds(selected_meshes), props)
    if estimate is not None and estimate[3] > 100_000_000:
        layout.label(text="Large grid", icon='ERROR')
    layout.operator("mesh.export_dicom", text=button_text, icon='EXPORT')


class VIEW3D_PT_dicomator_panel(Panel):
    """Root panel that hosts the add-on UI."""

    bl_label = "DICOMator"
    bl_idname = "VIEW3D_PT_dicomator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        props = context.scene.dicomator_props
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="Select a mesh object to export", icon='INFO')
            return
        else:
            selected_meshes = _selected_meshes(context)
            layout.label(text=_export_summary(selected_meshes), icon='MESH_DATA')

        grid = layout.grid_flow(columns=2, even_columns=True, even_rows=True, align=True)
        grid.prop(props, "export_image_series", text="Image")
        grid.prop(props, "export_drr", text="DRR")
        grid.prop(props, "export_rtdose", text="Dose")
        grid.prop(props, "export_rtstruct", text="Structures")
        _draw_export_action(layout, context)


class VIEW3D_PT_dicomator_selection_info(Panel):
    bl_label = "Estimate"
    bl_idname = "VIEW3D_PT_dicomator_selection_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        props = context.scene.dicomator_props
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="No mesh selected", icon='INFO')
            return

        selected_meshes = _selected_meshes(context)
        active_obj = context.active_object
        selection_count = len(selected_meshes)
        if selection_count > 1:
            layout.label(text=f"Selected: {selection_count} meshes (Active: {active_obj.name})", icon='MESH_DATA')
        else:
            layout.label(text=f"Selected: {active_obj.name}", icon='MESH_DATA')

        obj_width, obj_height, obj_depth = _selection_bounds(selected_meshes)

        col = layout.column(align=True)
        col.label(text=f"Size: {obj_width:.2f} x {obj_height:.2f} x {obj_depth:.2f} m")

        estimate = _grid_estimate((obj_width, obj_height, obj_depth), props)
        if estimate is not None:
            est_width, est_height, est_depth, total_voxels = estimate

            col.label(text=f"Est. Grid: {est_width} x {est_height} x {est_depth}")
            col.label(text=f"Total Voxels: {total_voxels:,}")

            memory_mb = (total_voxels * 2) / (1024 * 1024)
            col.label(text=f"Est. Memory: {memory_mb:.1f} MB")

            if total_voxels > 100_000_000:
                col.label(text="Grid too large!", icon='CANCEL')
            elif total_voxels > 50_000_000:
                col.label(text="Large grid - may be slow", icon='ERROR')

        if getattr(props, "export_drr", False):
            camera_obj = context.scene.camera
            detector_box = layout.column(align=True)
            if camera_obj and camera_obj.type == 'CAMERA':
                detector_box.label(text=f"Active Camera: {camera_obj.name}", icon='CAMERA_DATA')
                detector_width, detector_height = resolve_drr_detector_size(
                    context.scene,
                    resolution_scale=get_float_prop(props, "drr_resolution_scale", 1.0),
                )
                detector_box.label(text=f"Detector: {detector_width} x {detector_height} px", icon='IMAGE_DATA')
            else:
                detector_box.label(text="Set an active scene camera for DRR export", icon='ERROR')


class VIEW3D_PT_dicomator_per_object_hu(Panel):
    bl_label = "Objects"
    bl_idname = "VIEW3D_PT_dicomator_per_object_hu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="No mesh selected", icon='INFO')
            return
        selected_meshes = _selected_meshes(context)
        props = context.scene.dicomator_props

        layout.prop(props, "imaging_modality", text="Material Presets")

        for obj in selected_meshes:
            col = layout.column(align=True)
            col.label(text=obj.name, icon='MESH_DATA')

            # DICOM object type selector determines which pipeline this mesh
            # feeds into when exported.
            col.prop(obj, "dicomator_object_type", text="DICOM Type")

            obj_type = getattr(obj, "dicomator_object_type", "CT")

            if obj_type == "CT":
                row = col.row(align=True)
                row.prop(obj, "dicomator_material", text="Material")
                row.prop(obj, "dicomator_hu", text="HU")

            elif obj_type == "RTDOSE":
                col.prop(obj, "dicomator_dose", text="Dose (Gy)")

            elif obj_type == "RTSTRUCT":
                col.prop(obj, "dicomator_roi_type", text="ROI Type")


class VIEW3D_PT_dicomator_patient_info(Panel):
    bl_label = "Series"
    bl_idname = "VIEW3D_PT_dicomator_patient_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.dicomator_props
        layout.prop(props, "series_description", text="Description")
        layout.prop(props, "patient_name")
        layout.prop(props, "patient_id")
        layout.prop(props, "patient_sex")
        layout.prop(props, "patient_position", text="Position")


class VIEW3D_PT_dicomator_export_settings(Panel):
    bl_label = "Export"
    bl_idname = "VIEW3D_PT_dicomator_export_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.dicomator_props

        row = layout.row(align=True)
        row.prop(props, "lateral_resolution_mm", text="Lateral (mm)")
        row.prop(props, "axial_resolution_mm", text="Axial (mm)")

        if getattr(props, "export_drr", False):
            drr_box = layout.column(align=True)
            drr_box.prop(props, "drr_resolution_scale")
            camera_obj = context.scene.camera
            if camera_obj and camera_obj.type == 'CAMERA':
                detector_width, detector_height = resolve_drr_detector_size(
                    context.scene,
                    resolution_scale=get_float_prop(props, "drr_resolution_scale", 1.0),
                )
                drr_box.label(text=f"Active Camera: {camera_obj.name}", icon='CAMERA_DATA')
                drr_box.label(text=f"Detector: {detector_width} x {detector_height} px", icon='IMAGE_DATA')
            else:
                drr_box.label(text="No active scene camera", icon='ERROR')

        layout.prop(props, "apply_modifiers", text="Apply Modifiers")
        layout.prop(props, "export_directory")

        col = layout.column(align=True)
        col.prop(props, "export_4d")
        if props.export_4d:
            row = col.row(align=True)
            row.prop(props, "use_timeline_range")
            if props.use_timeline_range:
                row = col.row(align=True)
                row.label(text=f"Timeline: {context.scene.frame_start} to {context.scene.frame_end}", icon='TIME')
            else:
                row = col.row(align=True)
                row.prop(props, "frame_start")
                row.prop(props, "frame_end")
            col.prop(props, "frame_step")

        export_dir_val = get_str_prop(props, "export_directory", "")
        resolved_path = resolve_output_directory(export_dir_val)
        if resolved_path and export_dir_val.strip().startswith('//'):
            layout.label(text=f"Resolved: {resolved_path}", icon='FILE_FOLDER')

        # Show RT Dose settings when at least one selected mesh is typed RTDOSE.
        selected_meshes_all = _selected_meshes(context)
        if selected_meshes_all:
            layout.label(text=_export_summary(selected_meshes_all), icon='OUTLINER_COLLECTION')

        any_dose = any(getattr(obj, "dicomator_object_type", "CT") == "RTDOSE" for obj in selected_meshes_all)
        if any_dose:
            dose_box = layout.column(align=True)
            dose_box.prop(props, "dose_type", text="Dose Type")
            dose_box.prop(props, "dose_summation_type", text="Summation Type")

        modality = getattr(props, "imaging_modality", None)
        is_mri = modality in MRI_MODALITIES
        if getattr(props, "export_drr", False):
            note_box = layout.column(align=True)
            if is_mri:
                note_box.label(text="Use CT presets for DRR attenuation", icon='ERROR')


class VIEW3D_PT_dicomator_artifacts(Panel):
    bl_label = "Artifacts"
    bl_idname = "VIEW3D_PT_dicomator_artifacts"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_export_settings"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context: Context) -> bool:  # pragma: no cover - Blender UI code
        props = getattr(context.scene, "dicomator_props", None)
        return props is not None and bool(getattr(props, "export_image_series", True))

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.dicomator_props
        modality = getattr(props, "imaging_modality", None)
        is_mri = modality in MRI_MODALITIES

        layout.label(text="MRI artifacts" if is_mri else "CT artifacts", icon='SHADERFX')

        gaussian_box = layout.box()
        gaussian_box.prop(props, "enable_noise", text="Gaussian")
        if props.enable_noise:
            label = "Std. Dev." if is_mri else "Std. Dev. (HU)"
            gaussian_box.prop(props, "noise_std_dev_hu", text=label)

        if is_mri:
            bias_box = layout.box()
            bias_box.prop(props, "enable_bias_field", text="Bias Field")
            if props.enable_bias_field:
                bias_box.prop(props, "bias_field_strength")
                bias_box.prop(props, "bias_field_scale")
        else:
            partial_box = layout.box()
            partial_box.prop(props, "enable_partial_volume", text="Partial Volume")
            if props.enable_partial_volume:
                row = partial_box.row(align=True)
                row.prop(props, "partial_volume_kernel")
                row.prop(props, "partial_volume_iterations")
                partial_box.prop(props, "partial_volume_mix")

            metal_box = layout.box()
            metal_box.prop(props, "enable_metal_artifacts", text="Metal Streaks")
            if props.enable_metal_artifacts:
                row = metal_box.row(align=True)
                row.prop(props, "metal_intensity")
                row.prop(props, "metal_density_threshold")
                row = metal_box.row(align=True)
                row.prop(props, "metal_num_streaks")
                row.prop(props, "metal_falloff")

            ring_box = layout.box()
            ring_box.prop(props, "enable_ring_artifacts", text="Rings")
            if props.enable_ring_artifacts:
                ring_box.prop(props, "ring_intensity")
                row = ring_box.row(align=True)
                row.prop(props, "ring_random_radius")
                if not props.ring_random_radius:
                    row.prop(props, "ring_radius")
                row.prop(props, "ring_thickness")
                ring_box.prop(props, "ring_jitter")

            poisson_box = layout.box()
            poisson_box.prop(props, "enable_poisson_noise", text="Quantum Noise")
            if props.enable_poisson_noise:
                poisson_box.prop(props, "poisson_scale")

        motion_box = layout.box()
        motion_box.prop(props, "enable_motion_artifact", text="Motion")
        if props.enable_motion_artifact:
            row = motion_box.row(align=True)
            row.prop(props, "motion_blur_size")
            row.prop(props, "motion_axis")
            motion_box.prop(props, "motion_severity")


__all__ = [
    "VIEW3D_PT_dicomator_panel",
    "VIEW3D_PT_dicomator_selection_info",
    "VIEW3D_PT_dicomator_per_object_hu",
    "VIEW3D_PT_dicomator_patient_info",
    "VIEW3D_PT_dicomator_export_settings",
    "VIEW3D_PT_dicomator_artifacts",
]
