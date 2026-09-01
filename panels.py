"""UI panel definitions for the DICOMator add-on."""
from __future__ import annotations

import math
import os

import bpy
from bpy.types import Context, Panel
from mathutils import Vector

from .constants import (
    MAX_TOTAL_VOXELS,
    MRI_MODALITIES,
    UI_FEATURE_ARTIFACTS,
    UI_FEATURE_FOUR_D,
    UI_FEATURE_OBJECT_PRIORITY,
    UI_FEATURE_OVERSIZED_GRIDS,
    UI_FEATURE_RT_DOSE,
    UI_MODE_LABELS,
    ensure_pydicom_available,
    estimate_peak_memory_bytes_for_props,
    export_outputs_selectable,
    get_pydicom_error,
    grid_limits_exceeded,
    oversized_grids_allowed,
    resolve_export_outputs,
    suppressed_feature_labels,
    ui_feature_visible,
    ui_mode_of,
)
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
    lateral_mm = get_float_prop(props, "lateral_resolution_mm", 2.0)
    axial_mm = get_float_prop(props, "axial_resolution_mm", 2.0)
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


def _mode_label(props: bpy.types.PropertyGroup) -> str:
    """Return the human-readable name of the selected interface mode."""

    return UI_MODE_LABELS[ui_mode_of(props)]


def _draw_mode_notice(layout: bpy.types.UILayout, props: bpy.types.PropertyGroup) -> None:
    """Report settings that are switched on but held inactive by the mode.

    Hiding a control must not leave it quietly changing the exported data, so
    the panel names anything the current mode is suppressing.
    """

    suppressed = suppressed_feature_labels(props)
    if suppressed:
        layout.label(
            text=f"Inactive in {_mode_label(props)} mode: {', '.join(suppressed)}",
            icon='INFO',
        )


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
    if os.path.exists(export_dir) and not os.path.isdir(export_dir):
        layout.label(text="Export path must be a folder", icon='ERROR')
        return
    if os.path.isdir(export_dir):
        try:
            output_has_files = bool(os.listdir(export_dir))
        except OSError:
            layout.label(text="Cannot inspect export folder", icon='ERROR')
            return
        if output_has_files:
            layout.label(text="Choose a new or empty export folder", icon='ERROR')
            return

    unit_scale = float(
        getattr(getattr(context.scene, "unit_settings", None), "scale_length", 1.0) or 1.0
    )
    if not math.isclose(unit_scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
        layout.label(text="Scene Unit Scale must be 1.0", icon='ERROR')
        return

    if not (context.active_object and context.active_object.type == 'MESH'):
        layout.label(text="Select a mesh", icon='INFO')
        return

    # Outputs are resolved through the UI mode so the button agrees with what
    # the operator will actually write.
    outputs = resolve_export_outputs(props)
    if not any(outputs.values()):
        layout.label(text="Choose at least one output", icon='INFO')
        return

    selected_meshes = _selected_meshes(context)
    counts = _export_type_counts(selected_meshes)
    if (outputs["image_series"] or outputs["drr"]) and not counts["CT"]:
        layout.label(text="No image mesh selected", icon='ERROR')
        return
    if outputs["rtdose"] and not counts["RTDOSE"]:
        layout.label(text="No dose mesh selected", icon='ERROR')
        return
    if outputs["rtstruct"] and not counts["RTSTRUCT"]:
        layout.label(text="No structure mesh selected", icon='ERROR')
        return
    if outputs["drr"]:
        camera_obj = context.scene.camera
        if camera_obj is None or camera_obj.type != 'CAMERA':
            layout.label(text="Set a scene camera", icon='ERROR')
            return

    estimate = _grid_estimate(_selection_bounds(selected_meshes), props)
    # The estimated peak memory has to be part of this check: the export
    # operator aborts on it too, so leaving it out let the button look clear
    # for a grid that would be rejected the moment it was pressed.
    if estimate is not None and grid_limits_exceeded(
        *estimate[:3], estimate_peak_memory_bytes_for_props(estimate[3], props)
    ):
        if oversized_grids_allowed(props):
            layout.label(text="Oversized grid allowed", icon='ERROR')
        else:
            layout.label(text="Grid too large - export will abort", icon='ERROR')
    layout.operator("dicomator.export_dicom", text=button_text, icon='EXPORT')


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

        mode_row = layout.row(align=True)
        mode_row.prop(props, "ui_mode", expand=True)
        _draw_mode_notice(layout, props)

        layout.label(text="Synthetic research data only - not clinical", icon='ERROR')
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="Select a mesh object to export", icon='INFO')
            return

        selected_meshes = _selected_meshes(context)
        layout.label(text=_export_summary(selected_meshes), icon='MESH_DATA')

        if export_outputs_selectable(props):
            grid = layout.grid_flow(columns=2, even_columns=True, even_rows=True, align=True)
            grid.prop(props, "export_image_series", text="Image")
            grid.prop(props, "export_drr", text="DRR")
            grid.prop(props, "export_rtdose", text="Dose")
            grid.prop(props, "export_rtstruct", text="Structures")
        else:
            # Basic and Intermediate write a single image series; a dose or
            # structure mesh left over from Advanced would go unexported.
            counts = _export_type_counts(selected_meshes)
            ignored = counts["RTDOSE"] + counts["RTSTRUCT"]
            if ignored:
                layout.label(
                    text=f"{ignored} non-image mesh(es) not exported in {_mode_label(props)} mode",
                    icon='INFO',
                )
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

            memory_bytes = estimate_peak_memory_bytes_for_props(total_voxels, props)
            memory_mb = memory_bytes / (1024 * 1024)
            col.label(text=f"Conservative Peak Memory: {memory_mb:.1f} MB")

            if grid_limits_exceeded(est_width, est_height, est_depth, memory_bytes):
                if oversized_grids_allowed(props):
                    col.label(text="Oversized grid allowed - may exhaust memory", icon='ERROR')
                else:
                    col.label(text="Grid too large - export blocked", icon='CANCEL')
            elif total_voxels > MAX_TOTAL_VOXELS // 2:
                col.label(text="Large grid - may be slow", icon='ERROR')

        if resolve_export_outputs(props)["drr"]:
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

        show_object_types = export_outputs_selectable(props)
        show_priority = ui_feature_visible(props, UI_FEATURE_OBJECT_PRIORITY)

        for obj in selected_meshes:
            col = layout.column(align=True)
            col.label(text=obj.name, icon='MESH_DATA')

            # DICOM object type selector determines which pipeline this mesh
            # feeds into when exported. Modes that write only an image series
            # hide it, and say so for meshes typed in Advanced mode rather
            # than offering dose or ROI settings that would go unused.
            if show_object_types:
                col.prop(obj, "dicomator_object_type", text="DICOM Type")

            obj_type = getattr(obj, "dicomator_object_type", "CT")

            if obj_type == "CT":
                row = col.row(align=True)
                row.prop(obj, "dicomator_material", text="Material")
                row.prop(obj, "dicomator_hu", text="HU")
                if show_priority:
                    col.prop(obj, "dicomator_priority", text="Overlap Priority")

            elif not show_object_types:
                col.label(text=f"Not exported in {_mode_label(props)} mode", icon='INFO')

            elif obj_type == "RTDOSE":
                col.prop(obj, "dicomator_dose", text="Dose (Gy)")
                if show_priority:
                    col.prop(obj, "dicomator_priority", text="Overlap Priority")

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
        layout.prop(props, "patient_birth_date", text="Birth Date")
        layout.prop(props, "patient_sex")
        layout.prop(props, "patient_position", text="Position")
        layout.prop(props, "study_id")
        layout.prop(props, "accession_number")


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

        outputs = resolve_export_outputs(props)

        row = layout.row(align=True)
        row.prop(props, "lateral_resolution_mm", text="Lateral (mm)")
        row.prop(props, "axial_resolution_mm", text="Axial (mm)")

        if outputs["drr"]:
            drr_box = layout.column(align=True)
            drr_box.prop(props, "drr_resolution_scale")
            drr_box.prop(props, "drr_water_attenuation_m_inv")
            camera_obj = context.scene.camera
            if camera_obj and camera_obj.type == 'CAMERA':
                detector_width, detector_height = resolve_drr_detector_size(
                    context.scene,
                    resolution_scale=get_float_prop(props, "drr_resolution_scale", 1.0),
                )
                drr_box.label(text=f"Active Camera: {camera_obj.name}", icon='CAMERA_DATA')
                drr_box.label(text=f"Detector: {detector_width} x {detector_height} px", icon='IMAGE_DATA')
                if str(getattr(camera_obj.data, "type", "PERSP")).upper() != "ORTHO":
                    drr_box.label(text="Perspective camera: spatial tags omitted", icon='ERROR')
            else:
                drr_box.label(text="No active scene camera", icon='ERROR')

        layout.prop(props, "apply_modifiers", text="Apply Modifiers")
        if ui_feature_visible(props, UI_FEATURE_OVERSIZED_GRIDS):
            layout.prop(props, "allow_oversized_grids", text="Allow Oversized Grids")
        layout.prop(props, "export_directory")

        if ui_feature_visible(props, UI_FEATURE_FOUR_D):
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
        if any_dose and ui_feature_visible(props, UI_FEATURE_RT_DOSE):
            dose_box = layout.column(align=True)
            dose_box.prop(props, "dose_type", text="Dose Type")
            dose_box.prop(props, "dose_summation_type", text="Summation Type")
            dose_box.prop(props, "dose_accumulation", text="Dose Overlap")

        unit_scale = float(getattr(getattr(context.scene, "unit_settings", None), "scale_length", 1.0) or 1.0)
        layout.label(text="Patient axes: +X left, +Y posterior, +Z superior", icon='ORIENTATION_GLOBAL')
        if not math.isclose(unit_scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
            layout.label(text="Set Scene Unit Scale to 1.0 before export", icon='ERROR')

        modality = getattr(props, "imaging_modality", None)
        is_mri = modality in MRI_MODALITIES
        if outputs["drr"]:
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
        if props is None or not ui_feature_visible(props, UI_FEATURE_ARTIFACTS):
            return False
        return resolve_export_outputs(props)["image_series"]

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        props = context.scene.dicomator_props
        modality = getattr(props, "imaging_modality", None)
        is_mri = modality in MRI_MODALITIES

        layout.label(text="MRI artifacts" if is_mri else "CT artifacts", icon='SHADERFX')
        layout.prop(props, "artifact_seed")

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

            distortion_box = layout.box()
            distortion_box.prop(props, "enable_geometric_distortion", text="Geometric Distortion")
            if props.enable_geometric_distortion:
                distortion_box.prop(props, "geometric_gradient_strength")
                row = distortion_box.row(align=True)
                row.prop(props, "geometric_b0_shift")
                row.prop(props, "geometric_readout_axis", text="")
                distortion_box.prop(props, "geometric_b0_scale")

            gibbs_box = layout.box()
            gibbs_box.prop(props, "enable_gibbs_ringing", text="Gibbs Ringing")
            if props.enable_gibbs_ringing:
                gibbs_box.prop(props, "gibbs_strength")
                gibbs_box.prop(props, "gibbs_truncation")
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
