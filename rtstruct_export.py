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

import math
import os
from datetime import datetime
from typing import Generator, Optional, Sequence

import bmesh
import bpy

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


def _walk_loops(cut_edges: list) -> tuple[list[list[tuple[float, float, float]]], int]:
    """Convert a set of BMEdge objects from a bisect into ordered closed loops.

    Returns ``(loops, dropped_edge_count)``. Each loop is a list of
    ``(x, y, z)`` float tuples in Blender world-space metres. Every input edge
    is consumed exactly once: it either becomes part of a closed loop or is
    counted in ``dropped_edge_count``. Open chains (from non-watertight
    meshes) and degenerate single-edge intersections are discarded, since
    emitting them as ``CLOSED_PLANAR`` contours would be incorrect.

    Dangling chains are peeled off first (repeatedly removing degree-1
    vertices), so branch points such as a loop with a stray edge attached do
    not derail the loop walk. At crossings where multiple continuations
    remain (e.g. a figure-eight cross-section), the walk closes back to its
    start vertex when possible and otherwise picks the most collinear unused
    edge, keeping each lobe's edges together.

    The walk uses vertex/edge object-identity comparisons, which are valid
    while the bmesh remains alive.
    """
    if not cut_edges:
        return [], 0

    # Incidence map using BMVert objects as keys and edge lists as values.
    incident: dict = {}
    for edge in cut_edges:
        v0, v1 = edge.verts
        incident.setdefault(v0, []).append(edge)
        incident.setdefault(v1, []).append(edge)

    used: set[int] = set()
    dropped = 0

    def _other(edge, vert):
        v0, v1 = edge.verts
        return v1 if vert is v0 else v0

    def _unused(vert) -> list:
        return [edge for edge in incident.get(vert, []) if id(edge) not in used]

    def _direction(a, b) -> tuple[float, float, float]:
        dx = float(b.co.x) - float(a.co.x)
        dy = float(b.co.y) - float(a.co.y)
        dz = float(b.co.z) - float(a.co.z)
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm <= 0.0:
            return (0.0, 0.0, 0.0)
        return (dx / norm, dy / norm, dz / norm)

    # ------------------------------------------------------------------
    # Peel dangling chains: repeatedly consume edges at degree-1 vertices.
    # Whatever remains has minimum degree 2, so the loop walk below cannot
    # be lured into a dead end right next to a branch vertex.
    # ------------------------------------------------------------------
    pending = [vert for vert in incident if len(_unused(vert)) == 1]
    while pending:
        vert = pending.pop()
        chain = _unused(vert)
        if len(chain) != 1:
            continue
        edge = chain[0]
        used.add(id(edge))
        dropped += 1
        neighbor = _other(edge, vert)
        if len(_unused(neighbor)) == 1:
            pending.append(neighbor)

    loops: list[list[tuple[float, float, float]]] = []

    for seed in cut_edges:
        if id(seed) in used:
            continue
        used.add(id(seed))
        start, current = seed.verts
        path: list = [start, current]
        prev_dir = _direction(start, current)
        edges_in_path = 1
        closed = False

        while True:
            candidates = _unused(current)
            if not candidates:
                break

            closing_edge = None
            best_edge = None
            best_next = None
            best_dir = None
            best_dot = -2.0
            for edge in candidates:
                nxt = _other(edge, current)
                if nxt is start and len(path) >= 3:
                    closing_edge = edge
                    break
                direction = _direction(current, nxt)
                dot = (
                    prev_dir[0] * direction[0]
                    + prev_dir[1] * direction[1]
                    + prev_dir[2] * direction[2]
                )
                if dot > best_dot:
                    best_edge, best_next, best_dir, best_dot = edge, nxt, direction, dot

            if closing_edge is not None:
                used.add(id(closing_edge))
                edges_in_path += 1
                closed = True
                break
            if best_edge is None:
                break
            used.add(id(best_edge))
            edges_in_path += 1
            path.append(best_next)
            prev_dir = best_dir
            current = best_next

        if closed and len(path) >= 3:
            loops.append([(float(v.co.x), float(v.co.y), float(v.co.z)) for v in path])
        else:
            dropped += edges_in_path

    return loops, dropped


def _world_bmesh(
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
    *,
    apply_modifiers: bool,
) -> bmesh.types.BMesh:
    """Return a new world-space bmesh for ``obj`` (caller must ``free()``)."""
    if apply_modifiers and depsgraph is not None:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_data = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        world_matrix = obj_eval.matrix_world
    else:
        obj_eval = None
        mesh_data = obj.data
        world_matrix = obj.matrix_world

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh_data)
        bm.transform(world_matrix)
    except Exception:
        bm.free()
        raise
    finally:
        if obj_eval is not None:
            obj_eval.to_mesh_clear()
    return bm


def _slice_plane(
    bm_base: bmesh.types.BMesh, z_m: float
) -> tuple[list[list[tuple[float, float, float]]], int]:
    """Bisect ``bm_base`` at ``z_m``; return ``(loops, dropped_edge_count)``."""
    bm_slice = bm_base.copy()
    try:
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
        return _walk_loops(cut_edges)
    finally:
        bm_slice.free()


def _extract_contours_iter(
    obj: bpy.types.Object,
    z_positions_m: Sequence[float],
    depsgraph: bpy.types.Depsgraph,
    *,
    apply_modifiers: bool = True,
) -> Generator[tuple[int, int], None, tuple[dict[float, list[list[tuple[float, float, float]]]], int]]:
    """Generator variant of :func:`extract_contours_from_mesh`.

    Yields ``(planes_done, total_planes)`` after each bisected Z plane and
    returns ``(contours, dropped_edge_count)`` where the count totals the
    open/ambiguous cut edges discarded across all planes.
    """
    bm_base = _world_bmesh(obj, depsgraph, apply_modifiers=apply_modifiers)
    try:
        contours: dict[float, list[list[tuple[float, float, float]]]] = {}
        dropped_total = 0
        for index, z_m in enumerate(z_positions_m, start=1):
            loops, dropped = _slice_plane(bm_base, z_m)
            contours[float(z_m)] = loops
            dropped_total += dropped
            yield index, len(z_positions_m)
    finally:
        bm_base.free()
    return contours, dropped_total


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
    generator = _extract_contours_iter(
        obj, z_positions_m, depsgraph, apply_modifiers=apply_modifiers
    )
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            contours, _dropped = stop.value
            return contours


#: Plain-data ROI definition consumed by :func:`build_rtstruct_dataset`:
#: ``(roi_name, (r, g, b), roi_interpreted_type, contours_by_z)`` where
#: ``contours_by_z`` maps Z metres to lists of ``(x, y, z)`` metre loops.
RoiDefinition = tuple[
    str,
    tuple[int, int, int],
    str,
    dict[float, list[list[tuple[float, float, float]]]],
]


def build_rtstruct_dataset(
    roi_defs: Sequence[RoiDefinition],
    *,
    study_instance_uid: str,
    frame_of_reference_uid: str,
    series_instance_uid: str,
    sop_instance_uid: str,
    date_str: str,
    time_str: str,
    patient_name: str = "Anonymous",
    patient_id: str = "12345678",
    patient_sex: str = "M",
    series_description: str = "RT Structure Set from DICOMator",
    series_number: int = 1,
    bbox_min_z_m: float = 0.0,
    vz_m: float = 0.0,
    referenced_ct_series_instance_uid: Optional[str] = None,
    referenced_ct_sop_class_uid: Optional[str] = None,
    referenced_ct_sop_instance_uids: Optional[Sequence[str]] = None,
):
    """Build an RT Structure Set ``FileDataset`` from plain contour data.

    This is the Blender-free half of the RTSTRUCT export: it takes ROI
    definitions already extracted from mesh objects and assembles the pydicom
    dataset, so it can be exercised (and unit-tested) without ``bpy``.
    The caller is responsible for saving the returned dataset.
    """
    if not shared_constants.ensure_pydicom_available():
        raise RuntimeError("pydicom not available")

    pydicom = shared_constants.pydicom
    Dataset = shared_constants.Dataset
    FileDataset = shared_constants.FileDataset

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = RTSTRUCT_SOP_CLASS
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

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
    # StructureSetLabel has VR SH (max 16 characters); longer values are
    # non-conformant and trigger pydicom warnings.
    ds.StructureSetLabel = str(series_description)[:16]
    ds.StructureSetName = series_description
    ds.StructureSetDate = date_str
    ds.StructureSetTime = time_str

    # --- Referenced Frame of Reference Sequence ---
    ref_for_seq = Dataset()
    ref_for_seq.FrameOfReferenceUID = frame_of_reference_uid

    # Reference the co-exported CT series (if provided) so downstream TPS
    # tools can trace the structure set back to the images it was drawn
    # on.  RTReferencedSeriesSequence is Type 1 inside a study item, so
    # when no CT series was co-exported the RTReferencedStudySequence is
    # omitted entirely rather than written with an empty series sequence.
    if (
        referenced_ct_series_instance_uid
        and referenced_ct_sop_instance_uids
        and referenced_ct_sop_class_uid
    ):
        rt_ref_study = Dataset()
        rt_ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.2"  # RT Study
        rt_ref_study.ReferencedSOPInstanceUID = study_instance_uid

        rt_ref_series = Dataset()
        rt_ref_series.SeriesInstanceUID = referenced_ct_series_instance_uid
        contour_image_sequence: list = []
        for sop_uid in referenced_ct_sop_instance_uids:
            img_item = Dataset()
            img_item.ReferencedSOPClassUID = referenced_ct_sop_class_uid
            img_item.ReferencedSOPInstanceUID = sop_uid
            contour_image_sequence.append(img_item)
        rt_ref_series.ContourImageSequence = contour_image_sequence
        rt_ref_study.RTReferencedSeriesSequence = [rt_ref_series]
        ref_for_seq.RTReferencedStudySequence = [rt_ref_study]
    ds.ReferencedFrameOfReferenceSequence = [ref_for_seq]

    # --- Structure Set ROI Sequence ---
    roi_sequence = []
    for roi_number, (roi_name, _colour, _roi_type, _contours) in enumerate(roi_defs, start=1):
        roi_item = Dataset()
        roi_item.ROINumber = roi_number
        roi_item.ReferencedFrameOfReferenceUID = frame_of_reference_uid
        # ROIName has VR LO (max 64 characters).
        roi_item.ROIName = str(roi_name)[:64]
        roi_item.ROIGenerationAlgorithm = "MANUAL"
        roi_sequence.append(roi_item)
    ds.StructureSetROISequence = roi_sequence

    # Build a lookup from Z plane → referenced CT SOP Instance UID so we
    # can attach a ContourImageSequence to every emitted contour.
    ct_uids_by_slice: list[str] = (
        list(referenced_ct_sop_instance_uids)
        if referenced_ct_sop_instance_uids is not None
        else []
    )

    def _referenced_ct_sop_uid_for_z(z_m: float) -> Optional[str]:
        if not ct_uids_by_slice or vz_m <= 0.0:
            return None
        idx = int(round((float(z_m) - bbox_min_z_m) / vz_m - 0.5))
        if 0 <= idx < len(ct_uids_by_slice):
            return ct_uids_by_slice[idx]
        return None

    # --- ROI Contour Sequence ---
    roi_contour_sequence = []
    for roi_number, (_roi_name, colour, _roi_type, contours_by_z) in enumerate(roi_defs, start=1):
        r, g, b = colour

        roi_contour_item = Dataset()
        roi_contour_item.ReferencedROINumber = roi_number
        roi_contour_item.ROIDisplayColor = [int(r), int(g), int(b)]

        contour_sequence = []
        for z_m, loops in sorted(contours_by_z.items()):
            ref_sop_uid = _referenced_ct_sop_uid_for_z(z_m)
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
                if ref_sop_uid and referenced_ct_sop_class_uid:
                    img_ref = Dataset()
                    img_ref.ReferencedSOPClassUID = referenced_ct_sop_class_uid
                    img_ref.ReferencedSOPInstanceUID = ref_sop_uid
                    contour_item.ContourImageSequence = [img_ref]
                contour_sequence.append(contour_item)

        roi_contour_item.ContourSequence = contour_sequence
        roi_contour_sequence.append(roi_contour_item)

    ds.ROIContourSequence = roi_contour_sequence

    # --- RT ROI Observations Sequence ---
    observations_sequence = []
    for obs_number, (roi_name, _colour, roi_type, _contours) in enumerate(roi_defs, start=1):
        obs_item = Dataset()
        obs_item.ObservationNumber = obs_number
        obs_item.ReferencedROINumber = obs_number
        # ROIObservationLabel has VR SH (max 16 characters).
        obs_item.ROIObservationLabel = str(roi_name)[:16]
        obs_item.RTROIInterpretedType = str(roi_type or "OAR").upper()
        obs_item.ROIInterpreter = ""
        observations_sequence.append(obs_item)

    ds.RTROIObservationsSequence = observations_sequence

    return ds


def export_rtstruct_to_dicom(
    objects: Sequence[bpy.types.Object],
    bbox_min: "mathutils.Vector",  # noqa: F821 – mathutils present in Blender
    voxel_size: Sequence[float] | float,
    n_slices: int,
    output_dir: str,
    depsgraph: bpy.types.Depsgraph,
    **kwargs,
) -> dict[str, str]:
    """Write an RT Structure Set DICOM file for the supplied mesh objects.

    Blocking wrapper around :func:`export_rtstruct_to_dicom_iter`; see that
    function for parameter documentation.
    """
    generator = export_rtstruct_to_dicom_iter(
        objects, bbox_min, voxel_size, n_slices, output_dir, depsgraph, **kwargs
    )
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


def export_rtstruct_to_dicom_iter(
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
    referenced_ct_series_instance_uid: Optional[str] = None,
    referenced_ct_sop_class_uid: Optional[str] = None,
    referenced_ct_sop_instance_uids: Optional[Sequence[str]] = None,
) -> Generator[tuple[int, int], None, dict[str, str]]:
    """Write an RT Structure Set DICOM file for the supplied mesh objects.

    Generator variant: yields ``(planes_done, total_planes)`` while contours
    are extracted, then returns the result dict.

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

    generate_uid = shared_constants.generate_uid

    os.makedirs(output_dir, exist_ok=True)

    # Resolve the voxel Z dimension (only Z is needed for contour planes).
    if isinstance(voxel_size, (list, tuple)) and len(voxel_size) == 3:
        vz_m = float(voxel_size[2])
    else:
        vz_m = float(voxel_size)

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
    warnings: list[str] = []
    total_planes = max(1, len(objects) * len(z_positions_m))
    planes_done = 0
    for obj in objects:
        subtask = _extract_contours_iter(
            obj, z_positions_m, depsgraph, apply_modifiers=apply_modifiers
        )
        while True:
            try:
                next(subtask)
            except StopIteration as stop:
                contours, dropped = stop.value
                all_contours.append(contours)
                if dropped:
                    warnings.append(
                        f"{dropped} open/ambiguous cut edge(s) discarded while "
                        f"contouring '{obj.name}' (non-watertight mesh?)"
                    )
                break
            planes_done += 1
            if planes_done % 8 == 0:
                yield planes_done, total_planes
    yield planes_done, total_planes

    roi_defs: list[RoiDefinition] = [
        (
            obj.name,
            _roi_colour_from_object(obj, index),
            str(getattr(obj, "dicomator_roi_type", "OAR")).upper(),
            contours_by_z,
        )
        for index, (obj, contours_by_z) in enumerate(zip(objects, all_contours))
    ]

    try:
        ds = build_rtstruct_dataset(
            roi_defs,
            study_instance_uid=study_instance_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            series_instance_uid=series_instance_uid,
            sop_instance_uid=sop_instance_uid,
            date_str=date_str,
            time_str=time_str,
            patient_name=patient_name,
            patient_id=patient_id,
            patient_sex=patient_sex,
            series_description=series_description,
            series_number=series_number,
            bbox_min_z_m=bbox_min_z_m,
            vz_m=vz_m,
            referenced_ct_series_instance_uid=referenced_ct_series_instance_uid,
            referenced_ct_sop_class_uid=referenced_ct_sop_class_uid,
            referenced_ct_sop_instance_uids=referenced_ct_sop_instance_uids,
        )

        if phase_index is not None:
            filename = os.path.join(output_dir, f"Phase_{int(phase_index):03d}_RTStruct.dcm")
        else:
            filename = os.path.join(output_dir, "RTStruct.dcm")

        ds.save_as(filename, enforce_file_format=True)

    except Exception as exc:
        return {"error": f"Error saving RT Structure Set DICOM file: {exc}"}

    return {
        "success": f"Successfully exported RT Structure Set to {filename}",
        "warnings": warnings,
    }


__all__ = [
    "build_rtstruct_dataset",
    "extract_contours_from_mesh",
    "export_rtstruct_to_dicom",
    "export_rtstruct_to_dicom_iter",
]
