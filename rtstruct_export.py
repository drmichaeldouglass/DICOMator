"""RT Structure Set DICOM export helper for the DICOMator add-on.

Extracts closed planar contours from selected Blender mesh objects and writes
a DICOM RT Structure Set file (SOP class 1.2.840.10008.5.1.4.1.1.481.3).

Algorithm
---------
For each Z plane in the dose/CT grid, a bmesh copy of each structure mesh is
bisected using ``bmesh.ops.bisect_plane``.  The resulting cut edges are
graph-walked to recover ordered closed loops.  Each loop is stored as a
``CLOSED_PLANAR`` contour in the ``ROIContourSequence``.

Coordinate convention
---------------------
Blender world-space units are assumed to be metres.  All DICOM spatial
attributes are expressed in millimetres (multiply by 1000).  The voxel Z
positions passed as ``z_positions_m`` are in metres; the contour points
emitted into the DICOM file are in millimetres.

ROI colour
----------
Each ROI's display colour in the structure set is read from the **diffuse
colour** of the first material slot of the corresponding Blender object
(``obj.material_slots[0].material.diffuse_color``).  If the object has no
material, a deterministic colour is drawn from a clinical-style palette
indexed by the ROI number so that structures remain distinguishable.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Sequence

import bmesh
import bpy
import numpy as np

from . import constants as shared_constants
from .constants import RTSTRUCT_SOP_CLASS

# ---------------------------------------------------------------------------
# Default colour palette used when an object has no material assigned.
# Colours are 0-255 RGB tuples loosely inspired by common TPS conventions.
# ---------------------------------------------------------------------------
_DEFAULT_ROI_COLOURS: list[tuple[int, int, int]] = [
    (255, 0, 0),    # red    – GTV
    (255, 165, 0),  # orange – CTV
    (0, 0, 255),    # blue   – PTV
    (0, 200, 0),    # green  – OAR
    (255, 255, 0),  # yellow
    (0, 255, 255),  # cyan
    (255, 0, 255),  # magenta
    (128, 0, 0),    # dark red
    (0, 128, 0),    # dark green
    (0, 0, 128),    # dark blue
]


def _roi_colour_from_object(obj: bpy.types.Object, fallback_index: int) -> tuple[int, int, int]:
    """Return an RGB 0-255 tuple for ``obj``'s display colour in the RTSTRUCT.

    The colour is taken from the first material's ``diffuse_color`` if
    available, otherwise a deterministic palette entry is used.
    """
    if obj.material_slots and obj.material_slots[0].material is not None:
        mat = obj.material_slots[0].material
        r, g, b = mat.diffuse_color[:3]
        return (
            int(round(max(0.0, min(1.0, r)) * 255)),
            int(round(max(0.0, min(1.0, g)) * 255)),
            int(round(max(0.0, min(1.0, b)) * 255)),
        )
    return _DEFAULT_ROI_COLOURS[fallback_index % len(_DEFAULT_ROI_COLOURS)]


def _walk_loops(cut_edges: list) -> list[list[tuple[float, float, float]]]:
    """Convert a set of BMEdge objects from a bisect into ordered closed loops.

    Each loop is returned as a list of ``(x, y, z)`` float tuples in Blender
    world-space metres.  Only loops with three or more points are returned
    (degenerate single-edge intersections are discarded).

    The walk uses vertex object-identity comparisons, which are valid while
    the bmesh remains alive.
    """
    if not cut_edges:
        return []

    # Build adjacency map using BMVert objects as keys.
    adj: dict = {}
    for edge in cut_edges:
        v0, v1 = edge.verts
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    loops: list[list[tuple[float, float, float]]] = []
    unvisited: set = set(adj.keys())

    while unvisited:
        start = next(iter(unvisited))
        loop_verts: list = [start]
        unvisited.discard(start)
        prev = None
        current = start

        while True:
            # Prefer vertices not yet visited; stop if we reach the start
            # (closed loop) or a dead end.
            neighbors = [n for n in adj.get(current, []) if n is not prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            if nxt is start and len(loop_verts) >= 3:
                # Successfully closed the loop.
                break
            if nxt not in unvisited:
                # Either reached a dead end or a vertex belonging to another
                # loop sub-path — stop here.
                break
            loop_verts.append(nxt)
            unvisited.discard(nxt)
            prev = current
            current = nxt

        if len(loop_verts) >= 3:
            loops.append([(float(v.co.x), float(v.co.y), float(v.co.z)) for v in loop_verts])

    return loops


def extract_contours_from_mesh(
    obj: bpy.types.Object,
    z_positions_m: Sequence[float],
    depsgraph: bpy.types.Depsgraph,
    *,
    apply_modifiers: bool = True,
) -> dict[float, list[list[tuple[float, float, float]]]]:
    """Extract closed planar contours at multiple Z positions from ``obj``.

    Parameters
    ----------
    obj:
        A Blender mesh object.  Its world-space transform is applied to all
        vertex positions before slicing.
    z_positions_m:
        Sequence of Z coordinates in Blender world-space **metres** at which
        to extract contour cross-sections.
    depsgraph:
        Evaluated dependency graph used to resolve modifier stacks when
        ``apply_modifiers`` is ``True``.
    apply_modifiers:
        When ``True`` the evaluated (modifier-applied) mesh is used.
        When ``False`` the base mesh data is used instead.

    Returns
    -------
    dict
        Maps each Z position (in metres) to a (possibly empty) list of
        contour loops.  Each loop is an ordered list of
        ``(x_m, y_m, z_m)`` world-space metre tuples.
        The caller is responsible for converting positions to millimetres
        when writing DICOM attributes.
    """
    if apply_modifiers and depsgraph is not None:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_data = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        world_matrix = obj_eval.matrix_world
    else:
        mesh_data = obj.data
        world_matrix = obj.matrix_world

    # Build a single base bmesh in world space to avoid repeated transforms.
    bm_base = bmesh.new()
    bm_base.from_mesh(mesh_data)
    bm_base.transform(world_matrix)

    # Release the temporary evaluated mesh now that we have the bmesh.
    if apply_modifiers and depsgraph is not None:
        obj_eval.to_mesh_clear()

    contours: dict[float, list[list[tuple[float, float, float]]]] = {}

    for z_m in z_positions_m:
        bm_slice = bm_base.copy()
        bm_slice.verts.ensure_lookup_table()
        bm_slice.edges.ensure_lookup_table()
        bm_slice.faces.ensure_lookup_table()

        geom_all = list(bm_slice.verts) + list(bm_slice.edges) + list(bm_slice.faces)
        result = bmesh.ops.bisect_plane(
            bm_slice,
            geom=geom_all,
            dist=1e-5,
            plane_co=(0.0, 0.0, float(z_m)),
            plane_no=(0.0, 0.0, 1.0),
            clear_outer=False,
            clear_inner=False,
        )

        cut_edges = [elem for elem in result["geom_cut"] if isinstance(elem, bmesh.types.BMEdge)]
        loops = _walk_loops(cut_edges)

        contours[float(z_m)] = loops
        bm_slice.free()

    bm_base.free()
    return contours


def export_rtstruct_to_dicom(
    objects: Sequence[bpy.types.Object],
    bbox_min: "mathutils.Vector",  # noqa: F821 – mathutils present in Blender
    voxel_size: Sequence[float] | float,
    n_slices: int,
    output_dir: str,
    depsgraph: bpy.types.Depsgraph,
    *,
    patient_name: str = "Anonymous",
    patient_id: str = "12345678",
    patient_sex: str = "M",
    patient_position: str = "HFS",
    series_description: str = "RT Structure Set from DICOMator",
    apply_modifiers: bool = True,
    study_instance_uid: Optional[str] = None,
    frame_of_reference_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    series_number: int = 1,
    phase_index: Optional[int] = None,
) -> dict[str, str]:
    """Write an RT Structure Set DICOM file for the supplied mesh objects.

    Parameters
    ----------
    objects:
        Sequence of Blender mesh objects to export as ROIs.  Each object
        yields one ROI in the structure set.
    bbox_min:
        World-space origin of the shared voxel grid in metres.  Used to
        compute the Z positions of contour planes so that the structure set
        aligns with the co-exported CT or dose volume.
    voxel_size:
        Voxel edge lengths in metres; either a scalar or a 3-tuple
        ``(vx_m, vy_m, vz_m)``.
    n_slices:
        Number of axial slices (depth dimension of the voxel grid).  The
        contour planes are placed at the centre of each voxel:
        ``z = bbox_min_z + (i + 0.5) × vz`` for *i* in 0…n_slices-1.
    output_dir:
        Absolute path to the output directory.
    depsgraph:
        Evaluated dependency graph from the current Blender context.
    patient_name, patient_id, patient_sex, patient_position:
        Standard DICOM patient module attributes.
    series_description:
        Human-readable label placed in ``SeriesDescription`` /
        ``StructureSetLabel``.
    apply_modifiers:
        Whether to evaluate modifier stacks when extracting mesh geometry.
    study_instance_uid, frame_of_reference_uid, series_instance_uid:
        Pre-generated UIDs shared with the co-exported volumes.
    series_number:
        DICOM series number for the RT Structure Set object.
    phase_index:
        Optional 1-based phase index used in the output filename.

    Returns
    -------
    dict
        ``{'success': ...}`` on success or ``{'error': ...}`` on failure.
    """
    if not shared_constants.ensure_pydicom_available():
        return {"error": "pydicom not available"}

    pydicom = shared_constants.pydicom
    Dataset = shared_constants.Dataset
    FileDataset = shared_constants.FileDataset
    generate_uid = shared_constants.generate_uid

    os.makedirs(output_dir, exist_ok=True)

    # Resolve voxel dimensions.
    if isinstance(voxel_size, (list, tuple)) and len(voxel_size) == 3:
        vx_m, vy_m, vz_m = (float(v) for v in voxel_size)
    else:
        vx_m = vy_m = vz_m = float(voxel_size)

    # Z positions at the voxel centre for each slice (metres).
    bbox_min_z_m = float(bbox_min.z)
    z_positions_m: list[float] = [bbox_min_z_m + (i + 0.5) * vz_m for i in range(n_slices)]

    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")

    study_instance_uid = study_instance_uid or generate_uid()
    frame_of_reference_uid = frame_of_reference_uid or generate_uid()
    series_instance_uid = series_instance_uid or generate_uid()
    sop_instance_uid = generate_uid()

    # Extract contours for every object before building the DICOM dataset so
    # that bmesh operations are completed while we still hold references.
    all_contours: list[dict[float, list[list[tuple[float, float, float]]]]] = []
    for obj in objects:
        contours = extract_contours_from_mesh(
            obj,
            z_positions_m,
            depsgraph,
            apply_modifiers=apply_modifiers,
        )
        all_contours.append(contours)

    try:
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = RTSTRUCT_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # --- SOP common ---
        ds.SOPClassUID = RTSTRUCT_SOP_CLASS
        ds.SOPInstanceUID = sop_instance_uid

        # --- Patient module ---
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex = patient_sex

        # --- General study ---
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.StudyID = "1"
        ds.AccessionNumber = "1"
        ds.ReferringPhysicianName = ""

        # --- RT series ---
        ds.Modality = "RTSTRUCT"
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = int(series_number)
        ds.SeriesDescription = series_description
        ds.SeriesDate = date_str
        ds.SeriesTime = time_str
        ds.Manufacturer = "DICOMator"
        ds.InstitutionName = "Virtual Hospital"
        ds.StationName = "Blender"

        # --- Structure Set Identification ---
        ds.StructureSetLabel = series_description
        ds.StructureSetName = series_description
        ds.StructureSetDate = date_str
        ds.StructureSetTime = time_str

        # --- Referenced Frame of Reference Sequence ---
        ref_for_seq = Dataset()
        ref_for_seq.FrameOfReferenceUID = frame_of_reference_uid

        # RTReferencedStudySequence (references the shared study)
        rt_ref_study = Dataset()
        rt_ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.2"  # RT Study
        rt_ref_study.ReferencedSOPInstanceUID = study_instance_uid
        rt_ref_study.RTReferencedSeriesSequence = []
        ref_for_seq.RTReferencedStudySequence = [rt_ref_study]
        ds.ReferencedFrameOfReferenceSequence = [ref_for_seq]

        # --- Structure Set ROI Sequence ---
        roi_sequence = []
        for roi_number, obj in enumerate(objects, start=1):
            roi_item = Dataset()
            roi_item.ROINumber = roi_number
            roi_item.ReferencedFrameOfReferenceUID = frame_of_reference_uid
            roi_item.ROIName = obj.name
            roi_item.ROIGenerationAlgorithm = "MANUAL"
            roi_sequence.append(roi_item)
        ds.StructureSetROISequence = roi_sequence

        # --- ROI Contour Sequence ---
        roi_contour_sequence = []
        for roi_number, (obj, contours_by_z) in enumerate(
            zip(objects, all_contours), start=1
        ):
            r, g, b = _roi_colour_from_object(obj, roi_number - 1)

            roi_contour_item = Dataset()
            roi_contour_item.ReferencedROINumber = roi_number
            roi_contour_item.ROIDisplayColor = [r, g, b]

            contour_sequence = []
            for z_m, loops in sorted(contours_by_z.items()):
                for loop in loops:
                    if len(loop) < 3:
                        continue
                    # Flatten to [x1_mm, y1_mm, z1_mm, x2_mm, ...] in mm.
                    contour_data: list[float] = []
                    for (x_m, y_m, _z) in loop:
                        contour_data.append(round(x_m * 1000.0, 4))
                        contour_data.append(round(y_m * 1000.0, 4))
                        contour_data.append(round(z_m * 1000.0, 4))

                    contour_item = Dataset()
                    contour_item.ContourGeometricType = "CLOSED_PLANAR"
                    contour_item.NumberOfContourPoints = len(loop)
                    contour_item.ContourData = contour_data
                    contour_sequence.append(contour_item)

            roi_contour_item.ContourSequence = contour_sequence
            roi_contour_sequence.append(roi_contour_item)

        ds.ROIContourSequence = roi_contour_sequence

        # --- RT ROI Observations Sequence ---
        observations_sequence = []
        for obs_number, obj in enumerate(objects, start=1):
            obs_item = Dataset()
            obs_item.ObservationNumber = obs_number
            obs_item.ReferencedROINumber = obs_number
            obs_item.ROIObservationLabel = obj.name
            # Read the per-object ROI type; fall back to 'OAR' if unset.
            roi_type = str(getattr(obj, "dicomator_roi_type", "OAR")).upper()
            obs_item.RTROIInterpretedType = roi_type
            obs_item.ROIInterpreter = ""
            observations_sequence.append(obs_item)

        ds.RTROIObservationsSequence = observations_sequence

        if phase_index is not None:
            filename = os.path.join(output_dir, f"Phase_{int(phase_index):03d}_RTStruct.dcm")
        else:
            filename = os.path.join(output_dir, "RTStruct.dcm")

        ds.save_as(filename)

    except Exception as exc:
        return {"error": f"Error saving RT Structure Set DICOM file: {exc}"}

    return {"success": f"Successfully exported RT Structure Set to {filename}"}


__all__ = [
    "extract_contours_from_mesh",
    "export_rtstruct_to_dicom",
]
