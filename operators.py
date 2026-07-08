"""Operator definitions for the DICOMator add-on.

The export operator runs modally: the heavy pipeline stages (voxelization,
slice writing, DRR projection, contour extraction) are generators that yield
progress, and a window-manager timer drains them in small time slices so the
Blender UI stays responsive and the export can be cancelled with ESC.
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime
from functools import partial
from typing import Generator, Iterable, Sequence

import bmesh
import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

from .artifacts import (
    add_bias_field_shading,
    add_gaussian_noise,
    add_gibbs_ringing,
    add_metal_artifacts,
    add_motion_artifact,
    add_mri_geometric_distortion,
    add_poisson_noise,
    add_rician_noise,
    add_ring_artifacts,
    apply_partial_volume_effect,
)
from . import constants as shared_constants
from .constants import (
    AIR_DENSITY,
    MODALITY_CT,
    MRI_MODALITIES,
    ensure_pydicom_available,
)
from .dicom_export import export_projection_to_dicom, export_voxel_grid_to_dicom_iter
from .drr import generate_drr_from_hu_volume_iter
from .rtdose_export import export_rtdose_to_dicom
from .rtstruct_export import export_rtstruct_to_dicom_iter
from .utils import get_float_prop, resolve_output_directory
from .voxelization import (
    _world_vertex_array,
    prepare_object_geometry_iter,
    voxelize_objects_to_dose_iter,
    voxelize_objects_to_hu_iter,
)

#: Wall-clock budget (seconds) spent inside the export job per timer tick.
_MODAL_TIME_BUDGET_S = 0.1


def _get_int_prop(props, name: str, default: int) -> int:
    """Safely read an integer property, falling back to ``default`` on failure."""

    try:
        return int(getattr(props, name))
    except Exception:
        return int(default)


def _configured_artifact_stages(props) -> list:
    """Build the ordered artifact operations enabled in ``props``.

    Each entry is a callable taking and returning a volume; parameters are
    read from ``props`` once, when the list is built.
    """

    stages: list = []
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
        stages.append(partial(apply_partial_volume_effect, kernel_size=kernel, iterations=iterations, mix=mix))

    if getattr(props, "enable_metal_artifacts", False) and is_ct:
        intensity = max(0.0, get_float_prop(props, "metal_intensity", 400.0))
        threshold = get_float_prop(props, "metal_density_threshold", 2000.0)
        streaks = max(0, _get_int_prop(props, "metal_num_streaks", 10))
        falloff = max(0.1, get_float_prop(props, "metal_falloff", 6.0))
        stages.append(partial(
            add_metal_artifacts,
            intensity=float(intensity),
            density_threshold=float(threshold),
            num_streaks=streaks,
            falloff=float(falloff),
        ))

    if getattr(props, "enable_ring_artifacts", False) and is_ct:
        ring_intensity = max(0.0, get_float_prop(props, "ring_intensity", 80.0))
        ring_radius = None if getattr(props, "ring_random_radius", False) else get_float_prop(props, "ring_radius", 0.5)
        # ``add_ring_artifacts`` requires strictly positive thickness.
        # Keep UI-entered zero values from aborting the export pipeline.
        thickness = max(1e-4, get_float_prop(props, "ring_thickness", 0.02))
        jitter = max(0.0, get_float_prop(props, "ring_jitter", 0.02))
        stages.append(partial(
            add_ring_artifacts,
            ring_intensity=float(ring_intensity),
            ring_radius=float(ring_radius) if ring_radius is not None else None,
            thickness=float(thickness),
            jitter=float(jitter),
        ))

    if getattr(props, "enable_motion_artifact", False):
        blur_size = max(1, _get_int_prop(props, "motion_blur_size", 9))
        if blur_size % 2 == 0:
            blur_size += 1
        severity = max(0.0, min(1.0, get_float_prop(props, "motion_severity", 0.5)))
        axis_prop = getattr(props, "motion_axis", 'X')
        axis = 0 if str(axis_prop).upper() != 'Y' else 1
        stages.append(partial(
            add_motion_artifact,
            blur_size=blur_size,
            severity=float(severity),
            axis=axis,
        ))

    if getattr(props, "enable_geometric_distortion", False) and is_mri:
        gradient = get_float_prop(props, "geometric_gradient_strength", 0.05)
        b0 = max(0.0, get_float_prop(props, "geometric_b0_shift", 3.0))
        b0_scale = max(0.05, min(1.0, get_float_prop(props, "geometric_b0_scale", 0.35)))
        readout_axis = 0 if str(getattr(props, "geometric_readout_axis", 'Y')).upper() != 'Y' else 1
        stages.append(partial(
            add_mri_geometric_distortion,
            gradient_strength=float(gradient),
            b0_strength=float(b0),
            b0_scale=float(b0_scale),
            readout_axis=readout_axis,
        ))

    if getattr(props, "enable_gibbs_ringing", False) and is_mri:
        gibbs_strength = max(0.0, min(1.0, get_float_prop(props, "gibbs_strength", 0.6)))
        gibbs_truncation = max(0.0, min(0.49, get_float_prop(props, "gibbs_truncation", 0.2)))
        stages.append(partial(add_gibbs_ringing, strength=float(gibbs_strength), truncation=float(gibbs_truncation)))

    if getattr(props, "enable_bias_field", False) and is_mri:
        strength = max(0.0, min(1.0, get_float_prop(props, "bias_field_strength", 0.25)))
        scale = max(0.05, min(1.0, get_float_prop(props, "bias_field_scale", 0.3)))
        stages.append(partial(add_bias_field_shading, strength=float(strength), scale=float(scale)))

    if getattr(props, "enable_noise", False) and get_float_prop(props, "noise_std_dev_hu", 0.0) > 0.0:
        std_dev = max(0.0, get_float_prop(props, "noise_std_dev_hu", 20.0))
        if is_mri:
            # MR magnitude images carry Rician (not Gaussian) noise; the value is
            # reused as the underlying complex-channel standard deviation.
            stages.append(partial(add_rician_noise, sigma=float(std_dev)))
        else:
            stages.append(partial(add_gaussian_noise, std_hu=float(std_dev)))

    if getattr(props, "enable_poisson_noise", False) and get_float_prop(props, "poisson_scale", 0.0) > 0.0 and is_ct:
        scale = max(1.0, get_float_prop(props, "poisson_scale", 150.0))
        stages.append(partial(add_poisson_noise, scale=float(scale)))

    if is_mri and stages:
        # The artifact helpers clamp to the CT HU range, which permits
        # negative values; MR magnitude images are physically non-negative.
        stages.append(lambda volume: np.maximum(volume, 0))

    return stages


def _apply_configured_artifacts_iter(hu_array, props) -> Generator[tuple[int, int], None, object]:
    """Apply the configured artifacts, yielding ``(stage, total)`` progress.

    Yields between artifact stages so the modal operator can update the UI
    instead of running the whole chain inside a single timer tick.
    """

    stages = _configured_artifact_stages(props)
    total = max(1, len(stages))
    result = hu_array
    yield 0, total
    for index, stage in enumerate(stages, start=1):
        result = stage(result)
        yield index, total
    return result


def _apply_configured_artifacts(hu_array, props):
    """Apply the artifacts configured in ``props`` to ``hu_array`` sequentially."""

    generator = _apply_configured_artifacts_iter(hu_array, props)
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def _run_subtask(
    subtask: Generator[tuple[int, int], None, object],
    start: float,
    end: float,
) -> Generator[float, None, object]:
    """Drive a ``(current, total)``-yielding generator, re-yielding overall
    progress mapped into ``[start, end]``; returns the subtask's return value."""

    span = max(0.0, float(end) - float(start))
    while True:
        try:
            current, total = next(subtask)
        except StopIteration as stop:
            return stop.value
        fraction = min(1.0, max(0.0, float(current) / max(1.0, float(total))))
        yield float(start) + span * fraction


def _mesh_bounds_for_objects(
    objects: Sequence[bpy.types.Object],
    *,
    apply_modifiers: bool,
    depsgraph: bpy.types.Depsgraph | None,
) -> tuple[float, float, float, float, float, float]:
    """Return world-space bounds for the selected objects on the current frame."""

    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    found_vertex = False

    for obj in objects:
        if apply_modifiers and depsgraph is not None:
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            try:
                verts_world = _world_vertex_array(mesh, obj_eval.matrix_world)
            finally:
                obj_eval.to_mesh_clear()
            if verts_world.size:
                mins = verts_world.min(axis=0)
                maxs = verts_world.max(axis=0)
                min_x = min(min_x, float(mins[0]))
                max_x = max(max_x, float(maxs[0]))
                min_y = min(min_y, float(mins[1]))
                max_y = max(max_y, float(maxs[1]))
                min_z = min(min_z, float(mins[2]))
                max_z = max(max_z, float(maxs[2]))
                found_vertex = True
        else:
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                min_x = min(min_x, world_corner.x)
                max_x = max(max_x, world_corner.x)
                min_y = min(min_y, world_corner.y)
                max_y = max(max_y, world_corner.y)
                min_z = min(min_z, world_corner.z)
                max_z = max(max_z, world_corner.z)
                found_vertex = True

    if not found_vertex:
        raise ValueError("No valid mesh geometry found while estimating bounds")
    return min_x, max_x, min_y, max_y, min_z, max_z


def _bounds_across_frames_iter(
    context: bpy.types.Context,
    objects: Sequence[bpy.types.Object],
    frames: Iterable[int],
    *,
    apply_modifiers: bool,
) -> Generator[tuple[int, int], None, tuple[float, float, float, float, float, float]]:
    """Yield per-frame progress while gathering bounds across animation frames."""

    frames = list(frames)
    saved_frame = context.scene.frame_current
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    try:
        for index, frame in enumerate(frames, start=1):
            context.scene.frame_set(int(frame), subframe=0.0)
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
            yield index, len(frames)
    finally:
        context.scene.frame_set(saved_frame, subframe=0.0)

    return min_x, max_x, min_y, max_y, min_z, max_z


def _non_manifold_names_iter(
    objects: Sequence[bpy.types.Object],
) -> Generator[tuple[int, int], None, list[str]]:
    """Return names of objects whose base mesh has non-manifold edges.

    Ray-cast voxelization assumes watertight surfaces; non-manifold meshes can
    produce incorrectly filled columns. The check runs on the base mesh (a
    modifier stack may change manifoldness either way), so it is a heuristic
    warning rather than a guarantee.
    """

    names: list[str] = []
    for index, obj in enumerate(objects, start=1):
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            if any(not edge.is_manifold for edge in bm.edges):
                names.append(obj.name)
        finally:
            bm.free()
        yield index, len(objects)
    return names


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


def _series_description(base_description: str) -> str:
    """Return a series description appropriate for the requested output mode."""

    return str(base_description or "DICOMator Export").strip() or "DICOMator Export"


class MESH_OT_export_dicom(Operator):
    """Export selected meshes to enabled DICOM outputs (ESC cancels)."""

    bl_idname = "mesh.export_dicom"
    bl_label = "Export to DICOM"
    bl_options = {'REGISTER'}

    _timer = None
    _job = None
    # Class-level flag: only one modal export may run at a time.
    _running = False

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:  # pragma: no cover - Blender UI code
        return (
            not cls._running
            and context.active_object is not None
            and context.active_object.type == 'MESH'
            and context.active_object.mode == 'OBJECT'
        )

    def execute(self, context: bpy.types.Context):  # pragma: no cover - Blender runtime
        if MESH_OT_export_dicom._running:
            self.report({'ERROR'}, "A DICOM export is already running")
            return {'CANCELLED'}

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

        # ------------------------------------------------------------------
        # Partition selected meshes by their declared DICOM object type.
        # Objects whose dicomator_object_type is not set default to 'CT'
        # (backward-compatible with pre-RT versions of the add-on).
        # ------------------------------------------------------------------
        ct_objects = [obj for obj in selected_meshes if getattr(obj, "dicomator_object_type", "CT") == "CT"]
        dose_objects = [obj for obj in selected_meshes if getattr(obj, "dicomator_object_type", "CT") == "RTDOSE"]
        struct_objects = [obj for obj in selected_meshes if getattr(obj, "dicomator_object_type", "CT") == "RTSTRUCT"]

        if not ct_objects and not dose_objects and not struct_objects:
            self.report({'ERROR'}, "No selected meshes have a DICOM object type assigned")
            return {'CANCELLED'}

        props = context.scene.dicomator_props
        output_dir = resolve_output_directory(props.export_directory)
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

        export_image_series = bool(getattr(props, "export_image_series", True))
        export_drr = bool(getattr(props, "export_drr", False))
        export_rtdose = bool(getattr(props, "export_rtdose", False))
        export_rtstruct = bool(getattr(props, "export_rtstruct", False))

        if not (export_image_series or export_drr or export_rtdose or export_rtstruct):
            self.report({'ERROR'}, "Enable at least one DICOM output")
            return {'CANCELLED'}

        if (export_image_series or export_drr) and not ct_objects:
            self.report({'ERROR'}, "Image Series and DRR exports require at least one image-type mesh")
            return {'CANCELLED'}
        if export_rtdose and not dose_objects:
            self.report({'ERROR'}, "RT Dose export requires at least one RT Dose mesh")
            return {'CANCELLED'}
        if export_rtstruct and not struct_objects:
            self.report({'ERROR'}, "RT Structure export requires at least one RT Structure mesh")
            return {'CANCELLED'}

        export_objects = []
        if export_image_series or export_drr:
            export_objects.extend(ct_objects)
        if export_rtdose:
            export_objects.extend(dose_objects)
        if export_rtstruct:
            export_objects.extend(struct_objects)

        camera_obj = context.scene.camera if export_drr else None
        if export_drr and (camera_obj is None or camera_obj.type != 'CAMERA'):
            self.report({'ERROR'}, "Set an active scene camera before exporting a DRR")
            return {'CANCELLED'}

        lateral_mm = get_float_prop(props, "lateral_resolution_mm", 2.0)
        axial_mm = get_float_prop(props, "axial_resolution_mm", 2.0)
        if lateral_mm <= 0.0 or axial_mm <= 0.0:
            self.report({'ERROR'}, "Voxel spacing must be greater than zero")
            return {'CANCELLED'}

        modality_key = getattr(props, "imaging_modality", MODALITY_CT)
        if export_drr and modality_key in MRI_MODALITIES:
            self.report({'WARNING'}, "DRR uses current intensities as attenuation. CT modality presets are recommended.")

        try:
            frames = _frame_sequence(context, props)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        config = {
            'ct_objects': ct_objects,
            'dose_objects': dose_objects,
            'struct_objects': struct_objects,
            'export_objects': export_objects,
            'export_image_series': export_image_series,
            'export_drr': export_drr,
            'export_rtdose': export_rtdose,
            'export_rtstruct': export_rtstruct,
            'camera_obj': camera_obj,
            'output_dir': output_dir,
            'lateral_mm': float(lateral_mm),
            'axial_mm': float(axial_mm),
            'voxel_size_m': (float(lateral_mm) * 0.001, float(lateral_mm) * 0.001, float(axial_mm) * 0.001),
            'modality_key': modality_key,
            'dicom_modality': "MR" if modality_key in MRI_MODALITIES else "CT",
            'frames': frames,
            'apply_modifiers': bool(getattr(props, "apply_modifiers", True)),
            # One timestamp for the whole export so Study/Series/Content
            # dates and times agree across every co-exported object.
            'study_datetime': datetime.now(),
        }

        self._job = self._export_job(context, config)
        window_manager = context.window_manager
        window_manager.progress_begin(0, 1)
        window_manager.status_text_set("DICOMator: exporting... (ESC to cancel)")
        self._timer = window_manager.event_timer_add(0.02, window=context.window)
        window_manager.modal_handler_add(self)
        MESH_OT_export_dicom._running = True
        return {'RUNNING_MODAL'}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):  # pragma: no cover - Blender runtime
        if event.type == 'ESC':
            self._finish(context)
            self.report({'WARNING'}, "DICOM export cancelled")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        deadline = time.monotonic() + _MODAL_TIME_BUDGET_S
        progress = None
        try:
            while time.monotonic() < deadline:
                progress = next(self._job)
        except StopIteration as stop:
            self._finish(context)
            outcome = stop.value or {}
            if 'error' in outcome:
                self.report({'ERROR'}, outcome['error'])
                return {'CANCELLED'}
            self.report({'INFO'}, outcome.get('success', "DICOM export complete"))
            return {'FINISHED'}
        except Exception as exc:
            self._finish(context)
            self.report({'ERROR'}, f"Export failed: {exc}")
            return {'CANCELLED'}

        if progress is not None:
            context.window_manager.progress_update(float(progress))
        return {'RUNNING_MODAL'}

    def _finish(self, context: bpy.types.Context) -> None:  # pragma: no cover - Blender runtime
        MESH_OT_export_dicom._running = False
        window_manager = context.window_manager
        if self._timer is not None:
            window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._job is not None:
            # Closing the generator runs its finally blocks (frame restore).
            self._job.close()
            self._job = None
        window_manager.status_text_set(None)
        window_manager.progress_end()

    def _export_job(self, context: bpy.types.Context, config: dict) -> Generator[float, None, dict[str, str]]:
        """Generator that performs the full export, yielding 0..1 progress."""

        props = context.scene.dicomator_props
        frames: list[int] = config['frames']
        num_phases = len(frames)
        voxel_size_m: tuple[float, float, float] = config['voxel_size_m']
        apply_modifiers: bool = config['apply_modifiers']
        output_dir: str = config['output_dir']
        dicom_modality: str = config['dicom_modality']
        export_image_series: bool = config['export_image_series']
        export_drr: bool = config['export_drr']
        export_rtdose: bool = config['export_rtdose']
        export_rtstruct: bool = config['export_rtstruct']
        ct_objects = config['ct_objects']
        dose_objects = config['dose_objects']
        struct_objects = config['struct_objects']
        padding_voxels = 1

        # MR represents air as signal void (0), not as -1000 HU.
        background_value = 0.0 if dicom_modality == "MR" else AIR_DENSITY

        non_manifold = yield from _run_subtask(
            _non_manifold_names_iter(config['export_objects']), 0.0, 0.01
        )
        if non_manifold:
            shown = ", ".join(non_manifold[:5])
            suffix = "..." if len(non_manifold) > 5 else ""
            self.report(
                {'WARNING'},
                f"Non-manifold meshes may voxelize/contour incorrectly: {shown}{suffix}",
            )

        # ------------------------------------------------------------------
        # Compute a UNIFIED bounding box across enabled export objects
        # (CT + RTDOSE + RTSTRUCT as requested) so exported files share the
        # same coordinate space and FrameOfReferenceUID is meaningful.
        #
        # Single-phase exports evaluate each mesh exactly once here (BVH +
        # bounds) and reuse the prepared geometry during voxelization; 4D
        # exports must re-evaluate per frame, so they keep the bounds-only
        # sweep and rebuild geometry inside each phase.
        # ------------------------------------------------------------------
        prepared_geometry = None
        if num_phases == 1:
            frame_before_prepare = context.scene.frame_current
            context.scene.frame_set(int(frames[0]), subframe=0.0)
            try:
                prepared_geometry = yield from _run_subtask(
                    prepare_object_geometry_iter(
                        config['export_objects'],
                        context.evaluated_depsgraph_get(),
                        apply_modifiers=apply_modifiers,
                    ),
                    0.01,
                    0.03,
                )
            finally:
                context.scene.frame_set(frame_before_prepare, subframe=0.0)
            if not prepared_geometry:
                return {'error': "No valid mesh geometry found in the selected objects"}
            object_bounds = list(prepared_geometry.values())
            bounds = (
                min(geo[1][0] for geo in object_bounds),
                max(geo[1][1] for geo in object_bounds),
                min(geo[1][2] for geo in object_bounds),
                max(geo[1][3] for geo in object_bounds),
                min(geo[1][4] for geo in object_bounds),
                max(geo[1][5] for geo in object_bounds),
            )
        else:
            bounds = yield from _run_subtask(
                _bounds_across_frames_iter(
                    context,
                    config['export_objects'],
                    frames,
                    apply_modifiers=apply_modifiers,
                ),
                0.01,
                0.03,
            )
        padded_bounds = _pad_bounds(bounds, voxel_size_m, padding_voxels)
        estimated_width, estimated_height, estimated_depth = _estimate_grid_dimensions(padded_bounds, voxel_size_m)

        total_estimated_voxels = estimated_width * estimated_height * estimated_depth
        oversized = (
            estimated_width > 2000
            or estimated_height > 2000
            or estimated_depth > 2000
            or total_estimated_voxels > 100_000_000
        )
        if oversized:
            size_text = (
                f"{estimated_width}x{estimated_height}x{estimated_depth} "
                f"({total_estimated_voxels:,} voxels). "
                f"Selection size: {(bounds[1] - bounds[0]):.3f}x{(bounds[3] - bounds[2]):.3f}x{(bounds[5] - bounds[4]):.3f}m."
            )
            if not getattr(props, "allow_oversized_grids", False):
                return {
                    'error': (
                        f"Voxel grid too large: {size_text} "
                        "Limits: 2000 voxels per dimension, 100,000,000 total. "
                        "Increase the voxel spacing or enable 'Allow Oversized Grids' "
                        "in the Export panel."
                    )
                }
            self.report(
                {'WARNING'},
                (
                    f"Voxel grid very large: {size_text} "
                    "Continuing anyway (oversized grids allowed). "
                    "This may be slow or run out of memory."
                ),
            )

        series_description_base = _series_description(props.series_description)

        # Generate study-level UIDs once so enabled outputs belong to one
        # study and frame of reference.
        study_uid = shared_constants.generate_uid()
        frame_of_ref_uid = shared_constants.generate_uid()
        saved_frame = context.scene.frame_current

        # Build a human-readable list of active export types for reporting.
        active_types: list[str] = []
        if export_image_series:
            active_types.append(dicom_modality)
        if export_drr:
            active_types.append("DRR")
        if export_rtdose:
            active_types.append("RT Dose")
        if export_rtstruct:
            active_types.append("RT Structure")
        types_label = " + ".join(active_types)

        self.report(
            {'INFO'},
            (
                f"Exporting {num_phases} phase(s) [{types_label}] "
                f"lateral {config['lateral_mm']}mm / axial {config['axial_mm']}mm. "
                f"Grid: {estimated_width}x{estimated_height}x{estimated_depth}. "
                f"Shared StudyUID: {study_uid[:12]}... ForUID: {frame_of_ref_uid[:12]}..."
            ),
        )

        work_start = 0.03

        try:
            for phase_index, frame in enumerate(frames, start=1):
                phase_start = work_start + (1.0 - work_start) * (phase_index - 1) / num_phases
                phase_end = work_start + (1.0 - work_start) * phase_index / num_phases

                # Sub-divide the phase span across active export types.
                num_active = len(active_types)
                type_span = (phase_end - phase_start) / max(1, num_active)
                slot = 0

                context.scene.frame_set(frame, subframe=0.0)
                depsgraph = context.evaluated_depsgraph_get()
                yield phase_start

                phase_description = series_description_base
                if num_phases > 1:
                    # Label phases with their position in the respiratory/
                    # acquisition cycle starting at 0% (4DCT convention).
                    percent = ((phase_index - 1) / num_phases) * 100.0
                    phase_description = f"{series_description_base} - Phase {phase_index} ({percent:.1f}%)"

                # References collected from image export; RT Struct uses them
                # when an image series is exported in the same phase.
                ct_series_uid_for_struct: str | None = None
                ct_sop_class_uid_for_struct: str | None = None
                ct_sop_instance_uids_for_struct: list[str] = []

                # ----------------------------------------------------------
                # Image Series / DRR export from image-type meshes
                # ----------------------------------------------------------
                if ct_objects and (export_image_series or export_drr):
                    t_start = phase_start + slot * type_span
                    voxel_messages: list[str] = []
                    hu_array, origin, _dimensions = yield from _run_subtask(
                        voxelize_objects_to_hu_iter(
                            ct_objects,
                            voxel_size=voxel_size_m,
                            padding=0,
                            bbox_override=padded_bounds,
                            apply_modifiers=apply_modifiers,
                            depsgraph=depsgraph,
                            background_value=background_value,
                            messages=voxel_messages,
                            prepared=prepared_geometry,
                        ),
                        t_start,
                        t_start + type_span * 0.45,
                    )
                    for message in voxel_messages:
                        self.report({'WARNING'}, message)

                    if export_image_series:
                        write_start = phase_start + slot * type_span
                        slot += 1
                        image_series_uid = shared_constants.generate_uid()
                        hu_array_to_export = yield from _run_subtask(
                            _apply_configured_artifacts_iter(hu_array, props),
                            write_start + type_span * 0.45,
                            write_start + type_span * 0.55,
                        )
                        result = yield from _run_subtask(
                            export_voxel_grid_to_dicom_iter(
                                hu_array_to_export,
                                voxel_size_m,
                                output_dir,
                                origin,
                                patient_name=props.patient_name,
                                patient_id=props.patient_id,
                                patient_sex=props.patient_sex,
                                patient_birth_date=getattr(props, "patient_birth_date", ""),
                                series_description=phase_description,
                                direct_hu=True,
                                patient_position=props.patient_position,
                                dicom_modality=dicom_modality,
                                mr_weighting=config['modality_key'] if dicom_modality == "MR" else None,
                                study_id=getattr(props, "study_id", "1"),
                                accession_number=getattr(props, "accession_number", "1"),
                                study_datetime=config['study_datetime'],
                                study_instance_uid=study_uid,
                                frame_of_reference_uid=frame_of_ref_uid,
                                series_instance_uid=image_series_uid,
                                series_number=phase_index,
                                number_of_temporal_positions=num_phases if num_phases > 1 else None,
                                temporal_position_index=phase_index if num_phases > 1 else None,
                                temporal_position_identifier=phase_index if num_phases > 1 else None,
                                phase_index=phase_index if num_phases > 1 else None,
                            ),
                            write_start + type_span * 0.55,
                            write_start + type_span,
                        )
                        if 'error' in result:
                            return result

                        # Capture references for RTStruct to link back.
                        ct_series_uid_for_struct = image_series_uid
                        ct_sop_class_uid_for_struct = result.get('sop_class_uid')
                        ct_sop_instance_uids_for_struct = list(result.get('sop_instance_uids') or [])

                    if export_drr:
                        write_start = phase_start + slot * type_span
                        progress_start = write_start if export_image_series else write_start + type_span * 0.45
                        slot += 1
                        drr_series_uid = shared_constants.generate_uid()
                        projection_image, projection_metadata = yield from _run_subtask(
                            generate_drr_from_hu_volume_iter(
                                hu_array,
                                voxel_size_m,
                                origin,
                                context.scene,
                                config['camera_obj'],
                                resolution_scale=get_float_prop(props, "drr_resolution_scale", 1.0),
                                # Percentile auto-windowing varies per image;
                                # use the fixed physical mapping for 4D so
                                # intensities are comparable across phases.
                                fixed_normalization=num_phases > 1,
                            ),
                            progress_start,
                            write_start + type_span,
                        )
                        filename = (
                            f"Phase_{phase_index:03d}_DRR.dcm"
                            if num_phases > 1 else
                            "DRR_Image_0001.dcm"
                        )
                        drr_description = phase_description
                        if "DRR" not in drr_description.upper():
                            drr_description = f"{drr_description} - DRR"
                        result = export_projection_to_dicom(
                            projection_image,
                            output_dir,
                            filename=filename,
                            patient_name=props.patient_name,
                            patient_id=props.patient_id,
                            patient_sex=props.patient_sex,
                            patient_birth_date=getattr(props, "patient_birth_date", ""),
                            patient_position=props.patient_position,
                            series_description=drr_description,
                            study_id=getattr(props, "study_id", "1"),
                            accession_number=getattr(props, "accession_number", "1"),
                            study_datetime=config['study_datetime'],
                            pixel_spacing_mm=projection_metadata.get("pixel_spacing_mm"),
                            image_position_patient=projection_metadata.get("image_position_patient"),
                            image_orientation_patient=projection_metadata.get("image_orientation_patient"),
                            study_instance_uid=study_uid,
                            frame_of_reference_uid=frame_of_ref_uid,
                            series_instance_uid=drr_series_uid,
                            series_number=phase_index + (len(frames) if export_image_series else 0),
                            instance_number=1,
                            number_of_temporal_positions=num_phases if num_phases > 1 else None,
                            temporal_position_index=phase_index if num_phases > 1 else None,
                            temporal_position_identifier=phase_index if num_phases > 1 else None,
                        )
                        if 'error' in result:
                            return result

                # ----------------------------------------------------------
                # RT Dose export
                # ----------------------------------------------------------
                if export_rtdose and dose_objects:
                    t_start = phase_start + slot * type_span
                    slot += 1

                    dose_messages: list[str] = []
                    dose_array, dose_origin, _dose_dims = yield from _run_subtask(
                        voxelize_objects_to_dose_iter(
                            dose_objects,
                            voxel_size=voxel_size_m,
                            padding=0,
                            bbox_override=padded_bounds,
                            apply_modifiers=apply_modifiers,
                            depsgraph=depsgraph,
                            accumulate=getattr(props, "dose_accumulation", "SUM") == "SUM",
                            messages=dose_messages,
                            prepared=prepared_geometry,
                        ),
                        t_start,
                        t_start + type_span * 0.9,
                    )
                    for message in dose_messages:
                        self.report({'WARNING'}, message)

                    dose_series_uid = shared_constants.generate_uid()
                    result = export_rtdose_to_dicom(
                        dose_array,
                        voxel_size_m,
                        dose_origin,
                        output_dir,
                        patient_name=props.patient_name,
                        patient_id=props.patient_id,
                        patient_sex=props.patient_sex,
                        patient_birth_date=getattr(props, "patient_birth_date", ""),
                        patient_position=props.patient_position,
                        series_description=f"{phase_description} - RT Dose",
                        dose_type=getattr(props, "dose_type", "PHYSICAL"),
                        dose_summation_type=getattr(props, "dose_summation_type", "PLAN"),
                        study_id=getattr(props, "study_id", "1"),
                        accession_number=getattr(props, "accession_number", "1"),
                        study_datetime=config['study_datetime'],
                        study_instance_uid=study_uid,
                        frame_of_reference_uid=frame_of_ref_uid,
                        series_instance_uid=dose_series_uid,
                        series_number=phase_index + len(frames) * (int(export_image_series) + int(export_drr)),
                        phase_index=phase_index if num_phases > 1 else None,
                    )
                    if 'error' in result:
                        return result

                # ----------------------------------------------------------
                # RT Structure Set export
                # ----------------------------------------------------------
                if export_rtstruct and struct_objects:
                    t_start = phase_start + slot * type_span
                    slot += 1

                    struct_series_uid = shared_constants.generate_uid()
                    # bbox_min for RTSTRUCT uses the padded grid origin:
                    # padded_bounds = (min_x, max_x, min_y, max_y, min_z, max_z)
                    struct_bbox_min = Vector((padded_bounds[0], padded_bounds[2], padded_bounds[4]))
                    result = yield from _run_subtask(
                        export_rtstruct_to_dicom_iter(
                            struct_objects,
                            bbox_min=struct_bbox_min,
                            voxel_size=voxel_size_m,
                            n_slices=estimated_depth,
                            output_dir=output_dir,
                            depsgraph=depsgraph,
                            patient_name=props.patient_name,
                            patient_id=props.patient_id,
                            patient_sex=props.patient_sex,
                            patient_birth_date=getattr(props, "patient_birth_date", ""),
                            patient_position=props.patient_position,
                            series_description=f"{phase_description} - RT Structure",
                            study_id=getattr(props, "study_id", "1"),
                            accession_number=getattr(props, "accession_number", "1"),
                            study_datetime=config['study_datetime'],
                            apply_modifiers=apply_modifiers,
                            study_instance_uid=study_uid,
                            frame_of_reference_uid=frame_of_ref_uid,
                            series_instance_uid=struct_series_uid,
                            series_number=phase_index + len(frames) * (
                                int(export_image_series) + int(export_drr) + int(export_rtdose)
                            ),
                            phase_index=phase_index if num_phases > 1 else None,
                            referenced_ct_series_instance_uid=ct_series_uid_for_struct,
                            referenced_ct_sop_class_uid=ct_sop_class_uid_for_struct,
                            referenced_ct_sop_instance_uids=ct_sop_instance_uids_for_struct,
                        ),
                        t_start,
                        t_start + type_span,
                    )
                    if 'error' in result:
                        return result
                    for warning in result.get('warnings') or []:
                        self.report({'WARNING'}, warning)

                yield phase_end
        finally:
            context.scene.frame_set(saved_frame, subframe=0.0)

        return {'success': f"Successfully exported {num_phases} phase(s) [{types_label}] to {output_dir}"}


__all__ = ["MESH_OT_export_dicom"]
