"""Operator definitions for the DICOMator add-on."""
from __future__ import annotations

import math
import os

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

from .artifacts import (
    add_gaussian_noise,
    add_metal_artifacts,
    add_motion_artifact,
    add_poisson_noise,
    add_ring_artifacts,
    apply_partial_volume_effect,
)
from .constants import PYDICOM_AVAILABLE, generate_uid
from .drr import generate_drr_image
from .dicom_export import export_voxel_grid_to_dicom
from .utils import force_ui_redraw, get_float_prop
from .voxelization import voxelize_objects_to_hu

def _get_int_prop(props, name: str, default: int) -> int:
    """Safely read an integer property, falling back to ``default`` on failure."""

    try:
        return int(getattr(props, name))
    except Exception:
        return int(default)


def _apply_configured_artifacts(hu_array, props):
    """Apply the artifacts configured in ``props`` to ``hu_array`` sequentially."""

    result = hu_array

    if getattr(props, "enable_partial_volume", False):
        kernel = max(1, _get_int_prop(props, "partial_volume_kernel", 3))
        if kernel % 2 == 0:
            kernel += 1
        iterations = max(1, _get_int_prop(props, "partial_volume_iterations", 1))
        mix = max(0.0, min(1.0, get_float_prop(props, "partial_volume_mix", 1.0)))
        result = apply_partial_volume_effect(result, kernel_size=kernel, iterations=iterations, mix=mix)

    if getattr(props, "enable_metal_artifacts", False):
        intensity = max(0.0, get_float_prop(props, "metal_intensity", 400.0))
        threshold = get_float_prop(props, "metal_density_threshold", 2000.0)
        streaks = max(0, _get_int_prop(props, "metal_num_streaks", 10))
        falloff = max(0.1, get_float_prop(props, "metal_falloff", 6.0))
        result = add_metal_artifacts(
            result,
            intensity=float(intensity),
            density_threshold=float(threshold),
            num_streaks=streaks,
            falloff=float(falloff),
        )

    if getattr(props, "enable_ring_artifacts", False):
        ring_intensity = max(0.0, get_float_prop(props, "ring_intensity", 80.0))
        ring_radius = get_float_prop(props, "ring_radius", 0.5)
        thickness = max(0.0, get_float_prop(props, "ring_thickness", 0.02))
        jitter = max(0.0, get_float_prop(props, "ring_jitter", 0.02))
        result = add_ring_artifacts(
            result,
            ring_intensity=float(ring_intensity),
            ring_radius=float(ring_radius) if ring_radius is not None else None,
            thickness=float(thickness),
            jitter=float(jitter),
        )

    if getattr(props, "enable_motion_artifact", False):
        blur_size = max(1, _get_int_prop(props, "motion_blur_size", 9))
        if blur_size % 2 == 0:
            blur_size += 1
        severity = max(0.0, min(1.0, get_float_prop(props, "motion_severity", 0.5)))
        axis_prop = getattr(props, "motion_axis", 'X')
        axis = 0 if str(axis_prop).upper() != 'Y' else 1
        result = add_motion_artifact(
            result,
            blur_size=blur_size,
            severity=float(severity),
            axis=axis,
        )

    if getattr(props, "enable_noise", False) and get_float_prop(props, "noise_std_dev_hu", 0.0) > 0.0:
        std_dev = max(0.0, get_float_prop(props, "noise_std_dev_hu", 20.0))
        result = add_gaussian_noise(result, std_dev)

    if getattr(props, "enable_poisson_noise", False) and get_float_prop(props, "poisson_scale", 0.0) > 0.0:
        scale = max(1.0, get_float_prop(props, "poisson_scale", 150.0))
        result = add_poisson_noise(result, scale=scale)

    return result


def _save_drr_as_png(image_data: np.ndarray, filepath: str, scene: bpy.types.Scene | None = None) -> None:
    """Save a normalized 2D image array to ``filepath`` as an uncompressed PNG."""

    if image_data.ndim != 2:
        raise ValueError("Expected 2D image data for DRR export")

    height, width = image_data.shape
    if height <= 0 or width <= 0:
        raise ValueError("DRR image dimensions must be positive")

    image = bpy.data.images.new("DICOMator_DRR", width=width, height=height, alpha=True, float_buffer=False)
    try:
        normalized = np.clip(image_data.astype(np.float32), 0.0, 1.0)
        flipped = np.flipud(normalized)
        rgba = np.empty((height, width, 4), dtype=np.float32)
        rgba[..., 0] = flipped
        rgba[..., 1] = flipped
        rgba[..., 2] = flipped
        rgba[..., 3] = 1.0
        flat = np.ascontiguousarray(rgba.reshape(-1))
        image.pixels.foreach_set(flat)

        active_scene = scene or bpy.context.scene
        if active_scene is None:
            raise RuntimeError("No active scene available to save PNG")

        render_settings = active_scene.render.image_settings
        prev_format = render_settings.file_format
        prev_compression = render_settings.compression
        prev_color_depth = render_settings.color_depth
        try:
            render_settings.file_format = 'PNG'
            render_settings.compression = 0
            render_settings.color_depth = '8'
            image.file_format = 'PNG'
            image.filepath_raw = filepath
            image.save(filepath=filepath, scene=active_scene)
        finally:
            render_settings.file_format = prev_format
            render_settings.compression = prev_compression
            render_settings.color_depth = prev_color_depth
    finally:
        bpy.data.images.remove(image)


class MESH_OT_generate_drr(Operator):
    """Generate a digital radiograph reconstruction for selected meshes."""

    bl_idname = "mesh.generate_drr"
    bl_label = "Generate DRR"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:  # pragma: no cover - Blender UI code
        return (
            context.active_object is not None
            and context.active_object.type == 'MESH'
            and context.active_object.mode == 'OBJECT'
        )

    def execute(self, context: bpy.types.Context):  # pragma: no cover - Blender runtime
        props = context.scene.dicomator_props
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes and context.active_object and context.active_object.type == 'MESH':
            selected_meshes = [context.active_object]

        if not selected_meshes:
            self.report({'ERROR'}, "Please select at least one mesh object")
            return {'CANCELLED'}

        output_path_raw = getattr(props, "drr_output_path", "")
        if not output_path_raw or not str(output_path_raw).strip():
            self.report({'ERROR'}, "Please specify an output PNG file")
            return {'CANCELLED'}

        output_path = str(output_path_raw)
        if output_path.startswith('//'):
            relative_path = output_path[2:].replace('/', os.sep).replace('\\', os.sep)
            if bpy.data.filepath:
                base_dir = os.path.dirname(bpy.data.filepath)
            else:
                base_dir = os.getcwd()
            output_path = os.path.join(base_dir, relative_path)

        output_path = os.path.abspath(os.path.normpath(output_path))
        if not output_path.lower().endswith(".png"):
            output_path += ".png"

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as exc:
                self.report({'ERROR'}, f"Cannot create output directory: {exc}")
                return {'CANCELLED'}

        lateral_mm = get_float_prop(props, "lateral_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
        axial_mm = get_float_prop(props, "axial_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
        if lateral_mm <= 0.0 or axial_mm <= 0.0:
            self.report({'ERROR'}, "Voxel resolution must be positive")
            return {'CANCELLED'}

        angle_deg = get_float_prop(props, "drr_angle_deg", 0.0)
        apply_modifiers = bool(getattr(props, "apply_modifiers", True))
        depsgraph = context.evaluated_depsgraph_get() if apply_modifiers else None

        try:
            image_data = generate_drr_image(
                selected_meshes,
                lateral_resolution_mm=float(lateral_mm),
                axial_resolution_mm=float(axial_mm),
                angle_degrees=float(angle_deg),
                apply_modifiers=apply_modifiers,
                depsgraph=depsgraph,
            )
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to generate DRR: {exc}")
            return {'CANCELLED'}

        try:
            _save_drr_as_png(image_data, output_path, context.scene)
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to save DRR PNG: {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Saved DRR image to {output_path}")
        print(f"[DICOMator] DRR saved to {output_path}")
        return {'FINISHED'}


class MESH_OT_export_dicom(Operator):
    """Export selected meshes to a stack of DICOM files."""

    bl_idname = "mesh.export_dicom"
    bl_label = "Export to DICOM"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:  # pragma: no cover - Blender UI code
        return (
            context.active_object is not None
            and context.active_object.type == 'MESH'
            and context.active_object.mode == 'OBJECT'
        )

    def execute(self, context: bpy.types.Context):  # pragma: no cover - Blender runtime
        if not PYDICOM_AVAILABLE:
            self.report({'ERROR'}, "pydicom library not available")
            return {'CANCELLED'}

        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes:
            self.report({'ERROR'}, "Please select at least one mesh object")
            return {'CANCELLED'}

        active_obj = context.active_object
        if not active_obj or active_obj.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh object")
            return {'CANCELLED'}

        props = context.scene.dicomator_props
        output_dir = props.export_directory
        if not output_dir:
            self.report({'ERROR'}, "Please specify an export directory")
            return {'CANCELLED'}

        if output_dir.startswith('//'):
            relative_path = output_dir[2:].replace('/', os.sep).replace('\\', os.sep)
            if bpy.data.filepath:
                blend_dir = os.path.dirname(bpy.data.filepath)
                output_dir = os.path.join(blend_dir, relative_path)
            else:
                output_dir = os.path.join(os.getcwd(), relative_path)

        output_dir = os.path.abspath(os.path.normpath(output_dir))

        try:
            os.makedirs(output_dir, exist_ok=True)
            self.report({'INFO'}, f"Using export directory: {output_dir}")
        except Exception as exc:
            self.report({'ERROR'}, f"Cannot create output directory: {exc}")
            return {'CANCELLED'}

        if not os.access(output_dir, os.W_OK):
            self.report({'ERROR'}, f"Output directory is not writable: {output_dir}")
            return {'CANCELLED'}

        try:
            lateral_mm = get_float_prop(props, "lateral_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
            axial_mm = get_float_prop(props, "axial_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
            vx_m = float(lateral_mm) * 0.001
            vy_m = float(lateral_mm) * 0.001
            vz_m = float(axial_mm) * 0.001

            export_4d = bool(props.export_4d)
            apply_modifiers = bool(getattr(props, "apply_modifiers", True))
            if export_4d:
                if props.use_timeline_range:
                    start = int(context.scene.frame_start)
                    end = int(context.scene.frame_end)
                else:
                    start = int(props.frame_start)
                    end = int(props.frame_end)
                step = max(1, int(props.frame_step))
                if end < start:
                    self.report({'ERROR'}, "End frame must be >= start frame")
                    return {'CANCELLED'}

            padding = 1
            saved_frame = context.scene.frame_current
            try:
                if export_4d:
                    min_x = min_y = min_z = float('inf')
                    max_x = max_y = max_z = float('-inf')
                    for frame in range(start, end + 1, step):
                        context.scene.frame_set(frame, subframe=0.0)
                        context.scene.frame_current = frame
                        force_ui_redraw()
                        depsgraph = context.evaluated_depsgraph_get()
                        print(f"[DICOMator] Bounds pass at frame {frame}")
                        for obj in selected_meshes:
                            if apply_modifiers:
                                obj_eval = obj.evaluated_get(depsgraph)
                                mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
                                try:
                                    for vertex in mesh.vertices:
                                        world_vertex = obj_eval.matrix_world @ vertex.co
                                        min_x = min(min_x, world_vertex.x)
                                        max_x = max(max_x, world_vertex.x)
                                        min_y = min(min_y, world_vertex.y)
                                        max_y = max(max_y, world_vertex.y)
                                        min_z = min(min_z, world_vertex.z)
                                        max_z = max(max_z, world_vertex.z)
                                finally:
                                    obj_eval.to_mesh_clear()
                            else:
                                for corner in obj.bound_box:
                                    world_corner = obj.matrix_world @ Vector(corner)
                                    min_x = min(min_x, world_corner.x)
                                    max_x = max(max_x, world_corner.x)
                                    min_y = min(min_y, world_corner.y)
                                    max_y = max(max_y, world_corner.y)
                                    min_z = min(min_z, world_corner.z)
                                    max_z = max(max_z, world_corner.z)
                else:
                    depsgraph = context.evaluated_depsgraph_get()
                    min_x = min_y = min_z = float('inf')
                    max_x = max_y = max_z = float('-inf')
                    for obj in selected_meshes:
                        if apply_modifiers:
                            obj_eval = obj.evaluated_get(depsgraph)
                            mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
                            try:
                                for vertex in mesh.vertices:
                                    world_vertex = obj_eval.matrix_world @ vertex.co
                                    min_x = min(min_x, world_vertex.x)
                                    max_x = max(max_x, world_vertex.x)
                                    min_y = min(min_y, world_vertex.y)
                                    max_y = max(max_y, world_vertex.y)
                                    min_z = min(min_z, world_vertex.z)
                                    max_z = max(max_z, world_vertex.z)
                            finally:
                                obj_eval.to_mesh_clear()
                        else:
                            for corner in obj.bound_box:
                                world_corner = obj.matrix_world @ Vector(corner)
                                min_x = min(min_x, world_corner.x)
                                max_x = max(max_x, world_corner.x)
                                min_y = min(min_y, world_corner.y)
                                max_y = max(max_y, world_corner.y)
                                min_z = min(min_z, world_corner.z)
                                max_z = max(max_z, world_corner.z)
            finally:
                context.scene.frame_set(saved_frame)

            obj_width = max_x - min_x
            obj_height = max_y - min_y
            obj_depth = max_z - min_z

            estimated_width = int(math.ceil((obj_width + 2 * padding * vx_m) / vx_m))
            estimated_height = int(math.ceil((obj_height + 2 * padding * vy_m) / vy_m))
            estimated_depth = int(math.ceil((obj_depth + 2 * padding * vz_m) / vz_m))

            max_voxels_per_dimension = 2000
            total_estimated_voxels = estimated_width * estimated_height * estimated_depth
            max_total_voxels = 100_000_000

            if (
                estimated_width > max_voxels_per_dimension
                or estimated_height > max_voxels_per_dimension
                or estimated_depth > max_voxels_per_dimension
                or total_estimated_voxels > max_total_voxels
            ):
                self.report(
                    {'WARNING'},
                    (
                        "Voxel grid very large: "
                        f"{estimated_width}x{estimated_height}x{estimated_depth} "
                        f"({total_estimated_voxels:,} voxels). "
                        f"Selection size: {obj_width:.3f}x{obj_height:.3f}x{obj_depth:.3f}m. "
                        "Continuing anyway. This may be slow or run out of memory. "
                        f"Consider increasing resolution (current: lateral {lateral_mm}mm, axial {axial_mm}mm)."
                    ),
                )

            def progress_callback(current: int, total: int) -> None:
                progress = min(1.0, max(0.0, current / max(1, total)))
                context.window_manager.progress_update(progress)

            context.window_manager.progress_begin(0, 1)

            if not export_4d:
                self.report(
                    {'INFO'},
                    (
                        f"Voxelizing {len(selected_meshes)} mesh(es) with lateral {lateral_mm}mm, axial {axial_mm}mm. "
                        f"Estimated grid: {estimated_width}x{estimated_height}x{estimated_depth}"
                    ),
                )

                hu_array, origin, _dimensions = voxelize_objects_to_hu(
                    selected_meshes,
                    voxel_size=(vx_m, vy_m, vz_m),
                    padding=padding,
                    progress_callback=progress_callback,
                    apply_modifiers=apply_modifiers,
                    depsgraph=context.evaluated_depsgraph_get(),
                )

                hu_array_to_export = _apply_configured_artifacts(hu_array, props)

                result = export_voxel_grid_to_dicom(
                    hu_array_to_export,
                    (vx_m, vy_m, vz_m),
                    output_dir,
                    origin,
                    patient_name=props.patient_name,
                    patient_id=props.patient_id,
                    patient_sex=props.patient_sex,
                    series_description=props.series_description,
                    progress_callback=progress_callback,
                    direct_hu=True,
                    patient_position=props.patient_position,
                )
                context.window_manager.progress_end()
                if 'error' in result:
                    self.report({'ERROR'}, result['error'])
                    return {'CANCELLED'}
                self.report({'INFO'}, result['success'])
                return {'FINISHED'}

            frames = list(range(start, end + 1, step))
            num_phases = len(frames)
            self.report(
                {'INFO'},
                (
                    f"Exporting {num_phases} phase(s) with lateral {lateral_mm}mm, axial {axial_mm}mm. "
                    f"Grid: {estimated_width}x{estimated_height}x{estimated_depth}"
                ),
            )

            min_x_p = min_x - padding * vx_m
            max_x_p = max_x + padding * vx_m
            min_y_p = min_y - padding * vy_m
            max_y_p = max_y + padding * vy_m
            min_z_p = min_z - padding * vz_m
            max_z_p = max_z + padding * vz_m
            bbox_override = (min_x_p, max_x_p, min_y_p, max_y_p, min_z_p, max_z_p)

            study_uid = generate_uid()
            frame_of_ref_uid = generate_uid()

            saved_frame = context.scene.frame_current
            try:
                for phase_index, frame in enumerate(frames, start=1):
                    context.scene.frame_set(frame, subframe=0.0)
                    context.scene.frame_current = frame
                    force_ui_redraw()

                    hu_array, origin, _dimensions = voxelize_objects_to_hu(
                        selected_meshes,
                        voxel_size=(vx_m, vy_m, vz_m),
                        padding=0,
                        progress_callback=progress_callback,
                        bbox_override=bbox_override,
                        apply_modifiers=apply_modifiers,
                        depsgraph=context.evaluated_depsgraph_get(),
                    )

                    hu_array_to_export = _apply_configured_artifacts(hu_array, props)

                    percent = (phase_index / num_phases) * 100.0
                    phase_series_desc = f"{props.series_description} - Phase {phase_index} ({percent:.1f}%)"
                    series_uid = generate_uid()
                    series_number = phase_index

                    result = export_voxel_grid_to_dicom(
                        hu_array_to_export,
                        (vx_m, vy_m, vz_m),
                        output_dir,
                        origin,
                        patient_name=props.patient_name,
                        patient_id=props.patient_id,
                        patient_sex=props.patient_sex,
                        series_description=phase_series_desc,
                        progress_callback=progress_callback,
                        direct_hu=True,
                        patient_position=props.patient_position,
                        study_instance_uid=study_uid,
                        frame_of_reference_uid=frame_of_ref_uid,
                        series_instance_uid=series_uid,
                        series_number=series_number,
                        number_of_temporal_positions=num_phases,
                        temporal_position_index=frame,
                        temporal_position_identifier=phase_index,
                        phase_index=phase_index,
                    )
                    if 'error' in result:
                        context.window_manager.progress_end()
                        self.report({'ERROR'}, f"Phase {phase_index}: {result['error']}")
                        return {'CANCELLED'}
            finally:
                context.scene.frame_set(saved_frame)

            context.window_manager.progress_end()
            self.report({'INFO'}, f"Successfully exported {num_phases} phase(s) to {output_dir}")
            return {'FINISHED'}
        except Exception as exc:
            context.window_manager.progress_end()
            self.report({'ERROR'}, f"Export failed: {exc}")
            return {'CANCELLED'}


__all__ = ["MESH_OT_export_dicom"]
