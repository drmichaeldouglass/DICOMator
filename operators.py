"""Operator definitions for the DICOMator add-on."""
from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import bpy
from bpy.types import Operator
from mathutils import Vector

from .artifacts import (
    add_bias_field_shading,
    add_gaussian_noise,
    add_metal_artifacts,
    add_motion_artifact,
    add_poisson_noise,
    add_ring_artifacts,
    apply_partial_volume_effect,
)
from .constants import (
    MODALITY_CT,
    MRI_MODALITIES,
    OUTPUT_MODE_DRR,
    OUTPUT_MODE_VOLUME,
    ensure_pydicom_available,
    generate_uid,
)
from .dicom_export import export_projection_to_dicom, export_voxel_grid_to_dicom
from .drr import generate_drr_from_hu_volume
from .utils import force_ui_redraw, get_float_prop
from .voxelization import voxelize_objects_to_hu


def _get_int_prop(props, name: str, default: int) -> int:
    """Safely read an integer property, falling back to ``default`` on failure."""

    try:
        return int(getattr(props, name))
    except Exception:
        return int(default)


def _resolve_output_directory(output_dir: str) -> str:
    """Resolve Blender-relative paths into an absolute output directory."""

    resolved_dir = str(output_dir or "")
    if resolved_dir.startswith('//'):
        relative_path = resolved_dir[2:].replace('/', os.sep).replace('\\', os.sep)
        if bpy.data.filepath:
            blend_dir = os.path.dirname(bpy.data.filepath)
            resolved_dir = os.path.join(blend_dir, relative_path)
        else:
            resolved_dir = os.path.join(os.getcwd(), relative_path)
    return os.path.abspath(os.path.normpath(resolved_dir))


def _apply_configured_artifacts(hu_array, props):
    """Apply the artifacts configured in ``props`` to ``hu_array`` sequentially."""

    result = hu_array
    modality = getattr(props, "imaging_modality", MODALITY_CT)
    if modality not in MRI_MODALITIES and modality != MODALITY_CT:
        modality = MODALITY_CT
    is_mri = modality in MRI_MODALITIES
    is_ct = modality == MODALITY_CT

    if getattr(props, "enable_partial_volume", False) and is_ct:
        kernel = max(1, _get_int_prop(props, "partial_volume_kernel", 3))
        if kernel % 2 == 0:
            kernel += 1
        iterations = max(1, _get_int_prop(props, "partial_volume_iterations", 1))
        mix = max(0.0, min(1.0, get_float_prop(props, "partial_volume_mix", 1.0)))
        result = apply_partial_volume_effect(result, kernel_size=kernel, iterations=iterations, mix=mix)

    if getattr(props, "enable_metal_artifacts", False) and is_ct:
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

    if getattr(props, "enable_ring_artifacts", False) and is_ct:
        ring_intensity = max(0.0, get_float_prop(props, "ring_intensity", 80.0))
        ring_radius = None if getattr(props, "ring_random_radius", False) else get_float_prop(props, "ring_radius", 0.5)
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

    if getattr(props, "enable_bias_field", False) and is_mri:
        strength = max(0.0, min(1.0, get_float_prop(props, "bias_field_strength", 0.25)))
        scale = max(0.05, min(1.0, get_float_prop(props, "bias_field_scale", 0.3)))
        result = add_bias_field_shading(result, strength=float(strength), scale=float(scale))

    if getattr(props, "enable_noise", False) and get_float_prop(props, "noise_std_dev_hu", 0.0) > 0.0:
        std_dev = max(0.0, get_float_prop(props, "noise_std_dev_hu", 20.0))
        result = add_gaussian_noise(result, std_dev)

    if getattr(props, "enable_poisson_noise", False) and get_float_prop(props, "poisson_scale", 0.0) > 0.0 and is_ct:
        scale = max(1.0, get_float_prop(props, "poisson_scale", 150.0))
        result = add_poisson_noise(result, scale=scale)

    return result


def _make_progress_callback(context: bpy.types.Context, start: float, end: float):
    """Map a sub-task's 0..1 progress into the Blender progress bar."""

    span = max(0.0, float(end) - float(start))

    def callback(current: int, total: int) -> None:
        fraction = min(1.0, max(0.0, float(current) / max(1.0, float(total))))
        context.window_manager.progress_update(float(start) + span * fraction)

    return callback


def _mesh_bounds_for_objects(
    objects: Sequence[bpy.types.Object],
    *,
    apply_modifiers: bool,
    depsgraph: bpy.types.Depsgraph | None,
) -> tuple[float, float, float, float, float, float]:
    """Return world-space bounds for the selected objects on the current frame."""

    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for obj in objects:
        if apply_modifiers and depsgraph is not None:
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

    return min_x, max_x, min_y, max_y, min_z, max_z


def _bounds_across_frames(
    context: bpy.types.Context,
    objects: Sequence[bpy.types.Object],
    frames: Iterable[int],
    *,
    apply_modifiers: bool,
) -> tuple[float, float, float, float, float, float]:
    """Return world-space bounds spanning every requested animation frame."""

    saved_frame = context.scene.frame_current
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    try:
        for frame in frames:
            context.scene.frame_set(int(frame), subframe=0.0)
            force_ui_redraw()
            depsgraph = context.evaluated_depsgraph_get()
            frame_bounds = _mesh_bounds_for_objects(
                objects,
                apply_modifiers=apply_modifiers,
                depsgraph=depsgraph,
            )
            min_x = min(min_x, frame_bounds[0])
            max_x = max(max_x, frame_bounds[1])
            min_y = min(min_y, frame_bounds[2])
            max_y = max(max_y, frame_bounds[3])
            min_z = min(min_z, frame_bounds[4])
            max_z = max(max_z, frame_bounds[5])
    finally:
        context.scene.frame_set(saved_frame, subframe=0.0)

    return min_x, max_x, min_y, max_y, min_z, max_z


def _pad_bounds(
    bounds: tuple[float, float, float, float, float, float],
    voxel_size_m: tuple[float, float, float],
    padding_voxels: int,
) -> tuple[float, float, float, float, float, float]:
    """Expand bounds by ``padding_voxels`` in each direction."""

    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    vx_m, vy_m, vz_m = voxel_size_m
    return (
        min_x - padding_voxels * vx_m,
        max_x + padding_voxels * vx_m,
        min_y - padding_voxels * vy_m,
        max_y + padding_voxels * vy_m,
        min_z - padding_voxels * vz_m,
        max_z + padding_voxels * vz_m,
    )


def _estimate_grid_dimensions(
    bounds: tuple[float, float, float, float, float, float],
    voxel_size_m: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Estimate voxel grid dimensions from a padded world-space bounding box."""

    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    vx_m, vy_m, vz_m = voxel_size_m
    width = max(1, int(math.ceil((max_x - min_x) / vx_m)))
    height = max(1, int(math.ceil((max_y - min_y) / vy_m)))
    depth = max(1, int(math.ceil((max_z - min_z) / vz_m)))
    return width, height, depth


def _frame_sequence(context: bpy.types.Context, props) -> list[int]:
    """Return the frames selected for export."""

    if not bool(props.export_4d):
        return [int(context.scene.frame_current)]

    if props.use_timeline_range:
        start = int(context.scene.frame_start)
        end = int(context.scene.frame_end)
    else:
        start = int(props.frame_start)
        end = int(props.frame_end)
    step = max(1, int(props.frame_step))
    if end < start:
        raise ValueError("End frame must be >= start frame")
    return list(range(start, end + 1, step))


def _series_description(base_description: str, output_mode: str) -> str:
    """Return a series description appropriate for the requested output mode."""

    description = str(base_description or "DICOMator Export").strip() or "DICOMator Export"
    if output_mode == OUTPUT_MODE_DRR and "DRR" not in description.upper():
        return f"{description} - DRR"
    return description


class MESH_OT_export_dicom(Operator):
    """Export selected meshes to synthetic DICOM volumes or camera-based DRRs."""

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
        if not ensure_pydicom_available():
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
        output_dir = _resolve_output_directory(props.export_directory)
        if not output_dir:
            self.report({'ERROR'}, "Please specify an export directory")
            return {'CANCELLED'}

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as exc:
            self.report({'ERROR'}, f"Cannot create output directory: {exc}")
            return {'CANCELLED'}

        if not os.access(output_dir, os.W_OK):
            self.report({'ERROR'}, f"Output directory is not writable: {output_dir}")
            return {'CANCELLED'}

        output_mode = getattr(props, "output_mode", OUTPUT_MODE_VOLUME)
        camera_obj = context.scene.camera if output_mode == OUTPUT_MODE_DRR else None
        if output_mode == OUTPUT_MODE_DRR and (camera_obj is None or camera_obj.type != 'CAMERA'):
            self.report({'ERROR'}, "Set an active scene camera before exporting a DRR")
            return {'CANCELLED'}

        lateral_mm = get_float_prop(props, "lateral_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
        axial_mm = get_float_prop(props, "axial_resolution_mm", get_float_prop(props, "grid_resolution", 2.0))
        if lateral_mm <= 0.0 or axial_mm <= 0.0:
            self.report({'ERROR'}, "Voxel spacing must be greater than zero")
            return {'CANCELLED'}

        vx_m = float(lateral_mm) * 0.001
        vy_m = float(lateral_mm) * 0.001
        vz_m = float(axial_mm) * 0.001
        voxel_size_m = (vx_m, vy_m, vz_m)

        modality_key = getattr(props, "imaging_modality", MODALITY_CT)
        dicom_modality = "MR" if modality_key in MRI_MODALITIES else "CT"
        if output_mode == OUTPUT_MODE_DRR and modality_key in MRI_MODALITIES:
            self.report({'WARNING'}, "DRR uses current intensities as attenuation. CT modality presets are recommended.")

        try:
            frames = _frame_sequence(context, props)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        apply_modifiers = bool(getattr(props, "apply_modifiers", True))
        padding_voxels = 1

        try:
            bounds = _bounds_across_frames(
                context,
                selected_meshes,
                frames,
                apply_modifiers=apply_modifiers,
            )
            padded_bounds = _pad_bounds(bounds, voxel_size_m, padding_voxels)
            estimated_width, estimated_height, estimated_depth = _estimate_grid_dimensions(padded_bounds, voxel_size_m)

            total_estimated_voxels = estimated_width * estimated_height * estimated_depth
            if (
                estimated_width > 2000
                or estimated_height > 2000
                or estimated_depth > 2000
                or total_estimated_voxels > 100_000_000
            ):
                self.report(
                    {'WARNING'},
                    (
                        "Voxel grid very large: "
                        f"{estimated_width}x{estimated_height}x{estimated_depth} "
                        f"({total_estimated_voxels:,} voxels). "
                        f"Selection size: {(bounds[1] - bounds[0]):.3f}x{(bounds[3] - bounds[2]):.3f}x{(bounds[5] - bounds[4]):.3f}m. "
                        "Continuing anyway. This may be slow or run out of memory."
                    ),
                )

            num_phases = len(frames)
            series_description_base = _series_description(props.series_description, output_mode)
            study_uid = generate_uid()
            frame_of_ref_uid = generate_uid()
            saved_frame = context.scene.frame_current

            summary_label = "DRR projection(s)" if output_mode == OUTPUT_MODE_DRR else "synthetic series"
            self.report(
                {'INFO'},
                (
                    f"Exporting {num_phases} {summary_label} with lateral {lateral_mm}mm and axial {axial_mm}mm. "
                    f"Grid: {estimated_width}x{estimated_height}x{estimated_depth}"
                ),
            )

            context.window_manager.progress_begin(0, 1)

            try:
                for phase_index, frame in enumerate(frames, start=1):
                    phase_start = float(phase_index - 1) / float(num_phases)
                    phase_end = float(phase_index) / float(num_phases)
                    voxel_progress = _make_progress_callback(context, phase_start, phase_start + (phase_end - phase_start) * 0.55)
                    write_progress = _make_progress_callback(context, phase_start + (phase_end - phase_start) * 0.55, phase_end)

                    context.scene.frame_set(frame, subframe=0.0)
                    force_ui_redraw()

                    hu_array, origin, _dimensions = voxelize_objects_to_hu(
                        selected_meshes,
                        voxel_size=voxel_size_m,
                        padding=0,
                        progress_callback=voxel_progress,
                        bbox_override=padded_bounds,
                        apply_modifiers=apply_modifiers,
                        depsgraph=context.evaluated_depsgraph_get(),
                    )

                    phase_description = series_description_base
                    if num_phases > 1:
                        percent = (phase_index / num_phases) * 100.0
                        phase_description = f"{series_description_base} - Phase {phase_index} ({percent:.1f}%)"

                    series_uid = generate_uid()

                    if output_mode == OUTPUT_MODE_DRR:
                        projection_image, projection_metadata = generate_drr_from_hu_volume(
                            hu_array,
                            voxel_size_m,
                            origin,
                            context.scene,
                            camera_obj,
                            resolution_scale=get_float_prop(props, "drr_resolution_scale", 1.0),
                            progress_callback=write_progress,
                        )
                        filename = (
                            f"Phase_{phase_index:03d}_DRR.dcm"
                            if num_phases > 1 else
                            "DRR_Image_0001.dcm"
                        )
                        result = export_projection_to_dicom(
                            projection_image,
                            output_dir,
                            filename=filename,
                            patient_name=props.patient_name,
                            patient_id=props.patient_id,
                            patient_sex=props.patient_sex,
                            patient_position=props.patient_position,
                            series_description=phase_description,
                            pixel_spacing_mm=projection_metadata.get("pixel_spacing_mm"),
                            image_position_patient=projection_metadata.get("image_position_patient"),
                            image_orientation_patient=projection_metadata.get("image_orientation_patient"),
                            study_instance_uid=study_uid,
                            frame_of_reference_uid=frame_of_ref_uid,
                            series_instance_uid=series_uid,
                            series_number=phase_index,
                            instance_number=1,
                            number_of_temporal_positions=num_phases if num_phases > 1 else None,
                            temporal_position_index=phase_index if num_phases > 1 else None,
                            temporal_position_identifier=phase_index if num_phases > 1 else None,
                        )
                    else:
                        hu_array_to_export = _apply_configured_artifacts(hu_array, props)
                        result = export_voxel_grid_to_dicom(
                            hu_array_to_export,
                            voxel_size_m,
                            output_dir,
                            origin,
                            patient_name=props.patient_name,
                            patient_id=props.patient_id,
                            patient_sex=props.patient_sex,
                            series_description=phase_description,
                            progress_callback=write_progress,
                            direct_hu=True,
                            patient_position=props.patient_position,
                            dicom_modality=dicom_modality,
                            study_instance_uid=study_uid,
                            frame_of_reference_uid=frame_of_ref_uid,
                            series_instance_uid=series_uid,
                            series_number=phase_index,
                            number_of_temporal_positions=num_phases if num_phases > 1 else None,
                            temporal_position_index=phase_index if num_phases > 1 else None,
                            temporal_position_identifier=phase_index if num_phases > 1 else None,
                            phase_index=phase_index if num_phases > 1 else None,
                        )

                    if 'error' in result:
                        self.report({'ERROR'}, result['error'])
                        return {'CANCELLED'}
            finally:
                context.scene.frame_set(saved_frame, subframe=0.0)
                context.window_manager.progress_end()

            if output_mode == OUTPUT_MODE_DRR:
                self.report({'INFO'}, f"Successfully exported {num_phases} DRR projection(s) to {output_dir}")
            else:
                self.report({'INFO'}, f"Successfully exported {num_phases} synthetic volume phase(s) to {output_dir}")
            return {'FINISHED'}
        except Exception as exc:
            try:
                context.window_manager.progress_end()
            except Exception:
                pass
            self.report({'ERROR'}, f"Export failed: {exc}")
            return {'CANCELLED'}


__all__ = ["MESH_OT_export_dicom"]
