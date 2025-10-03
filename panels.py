"""UI panel definitions for the DICOMator add-on."""
from __future__ import annotations

import math
import os

import bpy
from bpy.types import Context, Panel
from mathutils import Vector

from .constants import MRI_MODALITIES, PYDICOM_AVAILABLE
from .utils import get_float_prop, get_str_prop


class VIEW3D_PT_dicomator_panel(Panel):
    """Root panel that hosts the add-on UI."""

    bl_label = "DICOMator"
    bl_idname = "VIEW3D_PT_dicomator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="Select a mesh object to export", icon='INFO')
        else:
            layout.label(text="Expand sections below to configure export", icon='TRIA_DOWN')


class VIEW3D_PT_dicomator_selection_info(Panel):
    bl_label = "Selection Info"
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

        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        active_obj = context.active_object
        # If nothing is selected but there is an active mesh, treat the active
        # object as the selection so bbox calculations have data.
        if not selected_meshes and active_obj and active_obj.type == 'MESH':
            selected_meshes = [active_obj]
        selection_count = len(selected_meshes)
        if selection_count > 1:
            layout.label(text=f"Selected: {selection_count} meshes (Active: {active_obj.name})", icon='MESH_DATA')
        else:
            layout.label(text=f"Selected: {active_obj.name}", icon='MESH_DATA')

        bbox_corners = []
        for obj in selected_meshes:
            bbox_corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
        min_x = min(corner.x for corner in bbox_corners)
        max_x = max(corner.x for corner in bbox_corners)
        min_y = min(corner.y for corner in bbox_corners)
        max_y = max(corner.y for corner in bbox_corners)
        min_z = min(corner.z for corner in bbox_corners)
        max_z = max(corner.z for corner in bbox_corners)

        obj_width = max_x - min_x
        obj_height = max_y - min_y
        obj_depth = max_z - min_z

        box = layout.box()
        box.label(text="Selection Info", icon='INFO')
        box.label(text=f"Size: {obj_width:.2f} x {obj_height:.2f} x {obj_depth:.2f} m")

        lateral_mm = get_float_prop(props, "lateral_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
        axial_mm = get_float_prop(props, "axial_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
        if lateral_mm > 0.0 and axial_mm > 0.0:
            vx = lateral_mm * 0.001
            vy = lateral_mm * 0.001
            vz = axial_mm * 0.001
            padding = 1
            est_width = int(math.ceil((obj_width + 2 * padding * vx) / vx))
            est_height = int(math.ceil((obj_height + 2 * padding * vy) / vy))
            est_depth = int(math.ceil((obj_depth + 2 * padding * vz) / vz))
            total_voxels = est_width * est_height * est_depth

            box.label(text=f"Est. Grid: {est_width} x {est_height} x {est_depth}")
            box.label(text=f"Total Voxels: {total_voxels:,}")

            memory_mb = (total_voxels * 2) / (1024 * 1024)
            box.label(text=f"Est. Memory: {memory_mb:.1f} MB")

            if total_voxels > 100_000_000:
                box.label(text="Grid too large!", icon='CANCEL')
            elif total_voxels > 50_000_000:
                box.label(text="Large grid - may be slow", icon='ERROR')


class VIEW3D_PT_dicomator_per_object_hu(Panel):
    bl_label = "Per-Object HU"
    bl_idname = "VIEW3D_PT_dicomator_per_object_hu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="No mesh selected", icon='INFO')
            return
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        props = context.scene.dicomator_props
        box_hu = layout.box()
        box_hu.label(text="Per-Object HU", icon='MOD_PHYSICS')
        box_hu.prop(props, "imaging_modality", text="Modality")
        for obj in selected_meshes:
            col = box_hu.column(align=True)
            col.label(text=obj.name, icon='MESH_DATA')
            row = col.row(align=True)
            row.prop(obj, "dicomator_material", text="Material")
            row.prop(obj, "dicomator_hu", text="Intensity")


class VIEW3D_PT_dicomator_patient_info(Panel):
    bl_label = "Patient Information"
    bl_idname = "VIEW3D_PT_dicomator_patient_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        props = context.scene.dicomator_props
        box = layout.box()
        box.label(text="Patient Information", icon='USER')
        box.prop(props, "patient_name")
        box.prop(props, "patient_id")
        box.prop(props, "patient_sex")


class VIEW3D_PT_dicomator_orientation(Panel):
    bl_label = "Image Orientation"
    bl_idname = "VIEW3D_PT_dicomator_orientation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        props = context.scene.dicomator_props
        box_or = layout.box()
        box_or.label(text="Image Orientation", icon='ORIENTATION_GIMBAL')
        box_or.prop(props, "patient_position", text="Patient Position")


class VIEW3D_PT_dicomator_export_settings(Panel):
    bl_label = "Export Settings"
    bl_idname = "VIEW3D_PT_dicomator_export_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: Context) -> None:  # pragma: no cover - Blender UI code
        layout = self.layout
        props = context.scene.dicomator_props

        box = layout.box()
        box.label(text="Export Settings", icon='SETTINGS')

        row = box.row(align=True)
        row.prop(props, "lateral_resolution_mm", text="Lateral (mm)")
        row.prop(props, "axial_resolution_mm", text="Axial (mm)")

        box.prop(props, "apply_modifiers", text="Apply Modifiers/Deformations")
        box.prop(props, "export_directory")

        col = box.column(align=True)
        col.prop(props, "export_4d")
        if props.export_4d:
            row = col.row(align=True)
            row.prop(props, "use_timeline_range")
            if props.use_timeline_range:
                row = col.row(align=True)
                row.label(text=f"Timeline: {context.scene.frame_start} → {context.scene.frame_end}", icon='TIME')
            else:
                row = col.row(align=True)
                row.prop(props, "frame_start")
                row.prop(props, "frame_end")
            col.prop(props, "frame_step")

        export_dir_val = get_str_prop(props, "export_directory", "")
        if export_dir_val.startswith('//'):
            relative_path = export_dir_val[2:].replace('/', os.sep).replace('\\', os.sep)
            if bpy.data.filepath:
                blend_dir = os.path.dirname(bpy.data.filepath)
                resolved_path = os.path.join(blend_dir, relative_path)
            else:
                resolved_path = os.path.join(os.getcwd(), relative_path)
            resolved_path = os.path.abspath(os.path.normpath(resolved_path))
            box.label(text=f"Resolved: {resolved_path}", icon='FILE_FOLDER')

        box.prop(props, "series_description")

        artifact_box = box.box()
        modality = getattr(props, "imaging_modality", None)
        is_mri = modality in MRI_MODALITIES
        artifact_box.label(text="MRI Artifacts" if is_mri else "CT Artifacts", icon='SHADERFX')

        gaussian_box = artifact_box.box()
        gaussian_box.label(text="Gaussian Noise", icon='RNDCURVE')
        gaussian_box.prop(props, "enable_noise")
        if props.enable_noise:
            label = "Std. Dev." if is_mri else "Std. Dev. (HU)"
            gaussian_box.prop(props, "noise_std_dev_hu", text=label)

        if is_mri:
            bias_box = artifact_box.box()
            bias_box.label(text="Bias Field Shading", icon='OUTLINER_OB_LIGHTPROBE')
            bias_box.prop(props, "enable_bias_field")
            if props.enable_bias_field:
                bias_box.prop(props, "bias_field_strength")
                bias_box.prop(props, "bias_field_scale")
        else:
            partial_box = artifact_box.box()
            partial_box.label(text="Partial Volume Blur", icon='MOD_SMOOTH')
            partial_box.prop(props, "enable_partial_volume")
            if props.enable_partial_volume:
                row = partial_box.row(align=True)
                row.prop(props, "partial_volume_kernel")
                row.prop(props, "partial_volume_iterations")
                partial_box.prop(props, "partial_volume_mix")

            metal_box = artifact_box.box()
            metal_box.label(text="Metal Streaks", icon='MOD_SIMPLIFY')
            metal_box.prop(props, "enable_metal_artifacts")
            if props.enable_metal_artifacts:
                row = metal_box.row(align=True)
                row.prop(props, "metal_intensity")
                row.prop(props, "metal_density_threshold")
                row = metal_box.row(align=True)
                row.prop(props, "metal_num_streaks")
                row.prop(props, "metal_falloff")

            ring_box = artifact_box.box()
            ring_box.label(text="Ring Artifacts", icon='MATSHADERBALL')
            ring_box.prop(props, "enable_ring_artifacts")
            if props.enable_ring_artifacts:
                ring_box.prop(props, "ring_intensity")
                row = ring_box.row(align=True)
                row.prop(props, "ring_radius")
                row.prop(props, "ring_thickness")
                ring_box.prop(props, "ring_jitter")

            poisson_box = artifact_box.box()
            poisson_box.label(text="Poisson Noise", icon='PARTICLES')
            poisson_box.prop(props, "enable_poisson_noise")
            if props.enable_poisson_noise:
                poisson_box.prop(props, "poisson_scale")

        motion_box = artifact_box.box()
        motion_box.label(text="Motion Blur", icon='ARROW_LEFTRIGHT')
        motion_box.prop(props, "enable_motion_artifact")
        if props.enable_motion_artifact:
            row = motion_box.row(align=True)
            row.prop(props, "motion_blur_size")
            row.prop(props, "motion_axis")
            motion_box.prop(props, "motion_severity")

        if PYDICOM_AVAILABLE:
            export_dir = get_str_prop(props, "export_directory", "")
            if export_dir.startswith('//'):
                relative_path = export_dir[2:].replace('/', os.sep).replace('\\', os.sep)
                if bpy.data.filepath:
                    blend_dir = os.path.dirname(bpy.data.filepath)
                    export_dir = os.path.join(blend_dir, relative_path)
                else:
                    export_dir = os.path.join(os.getcwd(), relative_path)
                export_dir = os.path.abspath(os.path.normpath(export_dir))

            if export_dir and export_dir.strip():
                if context.active_object and context.active_object.type == 'MESH':
                    selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
                    # If nothing is selected, fallback to the active mesh to
                    # ensure export UI can still compute bounds.
                    if not selected_meshes and context.active_object and context.active_object.type == 'MESH':
                        selected_meshes = [context.active_object]
                    bbox_corners = []
                    for obj in selected_meshes:
                        bbox_corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
                    min_x = min(corner.x for corner in bbox_corners)
                    max_x = max(corner.x for corner in bbox_corners)
                    min_y = min(corner.y for corner in bbox_corners)
                    max_y = max(corner.y for corner in bbox_corners)
                    min_z = min(corner.z for corner in bbox_corners)
                    max_z = max(corner.z for corner in bbox_corners)
                    obj_width = max_x - min_x
                    obj_height = max_y - min_y
                    obj_depth = max_z - min_z

                    lateral_mm_est = get_float_prop(props, "lateral_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
                    axial_mm_est = get_float_prop(props, "axial_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
                    if lateral_mm_est > 0.0 and axial_mm_est > 0.0:
                        vx = lateral_mm_est * 0.001
                        vy = lateral_mm_est * 0.001
                        vz = axial_mm_est * 0.001
                        est_total = (
                            int(math.ceil((obj_width + 2 * vx) / vx))
                            * int(math.ceil((obj_height + 2 * vy) / vy))
                            * int(math.ceil((obj_depth + 2 * vz) / vz))
                        )

                        if est_total > 100_000_000:
                            layout.label(text="Grid very large - may be slow or fail (memory)", icon='ERROR')
                        layout.operator("mesh.export_dicom", text="Export to DICOM", icon='EXPORT')
                    else:
                        layout.operator("mesh.export_dicom", text="Export to DICOM", icon='EXPORT')
                else:
                    layout.label(text="Select a mesh object to export", icon='INFO')
            else:
                layout.label(text="Please select export directory", icon='INFO')
        else:
            layout.label(text="pydicom library not available", icon='ERROR')


__all__ = [
    "VIEW3D_PT_dicomator_panel",
    "VIEW3D_PT_dicomator_selection_info",
    "VIEW3D_PT_dicomator_per_object_hu",
    "VIEW3D_PT_dicomator_patient_info",
    "VIEW3D_PT_dicomator_orientation",
    "VIEW3D_PT_dicomator_export_settings",
]
