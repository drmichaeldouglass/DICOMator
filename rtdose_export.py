"""RT Dose DICOM export helper for the DICOMator add-on.

Writes a dose grid (3-D float32 NumPy array in Gy) as a single multi-frame
DICOM RT Dose file (SOP class 1.2.840.10008.5.1.4.1.1.481.2).

Physical description
--------------------
The DICOM RT Dose storage format encodes dose as 32-bit unsigned integers
scaled by a ``DoseGridScaling`` factor:

    actual_dose_Gy = pixel_value × DoseGridScaling

The scaling factor is chosen so that the maximum dose in the grid maps to the
maximum uint32 value (4 294 967 295), giving the highest possible precision
across the full dynamic range of the grid.  If the grid is entirely zero the
scaling defaults to 1.0 Gy/count so the file is still valid.

The pixel data is a contiguous sequence of frames (slices) in inferior-to-
superior order, each frame stored as uint32 in row-major (C) order, matching
the ``rows × columns`` dimensions declared in the DICOM header.

Coordinate convention
---------------------
Spatial coordinates follow the same metres-to-mm conversion used throughout
DICOMator: Blender world-space units are assumed to be metres.  All DICOM
spatial attributes (``ImagePositionPatient``, ``PixelSpacing``,
``GridFrameOffsetVector``) are expressed in millimetres.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
from mathutils import Vector

from . import constants as shared_constants
from .constants import RTDOSE_SOP_CLASS


def export_rtdose_to_dicom(
    dose_grid: np.ndarray,
    voxel_size: Sequence[float] | float,
    bbox_min: Vector,
    output_dir: str,
    *,
    patient_name: str = "Anonymous",
    patient_id: str = "12345678",
    patient_sex: str = "M",
    patient_position: str = "HFS",
    series_description: str = "RT Dose from DICOMator",
    dose_type: str = "PHYSICAL",
    dose_summation_type: str = "PLAN",
    study_instance_uid: Optional[str] = None,
    frame_of_reference_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    series_number: int = 1,
    phase_index: Optional[int] = None,
) -> dict[str, str]:
    """Write ``dose_grid`` to a single multi-frame DICOM RT Dose file.

    Parameters
    ----------
    dose_grid:
        Float32 array with shape ``(W, H, D)`` — width (X), height (Y),
        depth (Z slices) — in Gy.  Background voxels should be 0.0 Gy.
    voxel_size:
        Voxel edge lengths in metres; either a scalar or a 3-tuple
        ``(vx_m, vy_m, vz_m)``.
    bbox_min:
        World-space origin of the voxel grid in metres
        (``mathutils.Vector``).
    output_dir:
        Absolute path to the output directory.  Created if it does not
        exist.
    patient_name, patient_id, patient_sex, patient_position:
        Standard DICOM patient module attributes.
    series_description:
        Human-readable label placed in ``SeriesDescription``.
    dose_type:
        DICOM ``DoseType`` attribute — ``'PHYSICAL'`` or ``'EFFECTIVE'``.
    dose_summation_type:
        DICOM ``DoseSummationType`` attribute — ``'PLAN'``, ``'FRACTION'``,
        or ``'BEAM'``.
    study_instance_uid, frame_of_reference_uid, series_instance_uid:
        Pre-generated UIDs to share with a co-exported CT or RT Structure
        Set.  New UIDs are generated if *None*.
    series_number:
        DICOM series number for this RT Dose object.
    phase_index:
        Optional 1-based phase index used in the output filename when
        exporting a 4D sequence.

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

    if isinstance(voxel_size, (list, tuple)) and len(voxel_size) == 3:
        vx_m, vy_m, vz_m = (float(v) for v in voxel_size)
    else:
        vx_m = vy_m = vz_m = float(voxel_size)

    vx_mm = vx_m * 1000.0
    vy_mm = vy_m * 1000.0
    vz_mm = vz_m * 1000.0

    bbox_min_mm = Vector((bbox_min.x * 1000.0, bbox_min.y * 1000.0, bbox_min.z * 1000.0))

    # Ensure we have a float32 array in (W, H, D) order; clamp negatives.
    dose_f32 = np.clip(np.asarray(dose_grid, dtype=np.float32), 0.0, None)

    width, height, depth = dose_f32.shape  # (W, H, D)

    # Compute the linear scaling factor.  Each pixel encodes:
    #   actual_dose = pixel_value × DoseGridScaling
    max_dose = float(dose_f32.max())
    uint32_max = float(np.iinfo(np.uint32).max)  # 4 294 967 295
    if max_dose > 0.0:
        dose_grid_scaling = max_dose / uint32_max
    else:
        # Grid is all-zero; use 1.0 Gy/count as a nominal scale so the
        # file round-trips correctly (all pixels remain zero).
        dose_grid_scaling = 1.0

    # Encode all frames into a (D × H × W) uint32 array.
    # Each frame is a 2D slice dose_f32[:, :, iz] with shape (W, H).
    # Transposing to (H, W) matches DICOM row-major convention (rows first).
    pixel_frames = np.zeros((depth, height, width), dtype=np.uint32)
    uint32_max_value = np.iinfo(np.uint32).max
    for iz in range(depth):
        slice_2d = dose_f32[:, :, iz]  # (W, H)
        # Clip before casting: float32 precision on a uint32-range scale can
        # push values a fraction above uint32_max, which silently wraps on
        # cast. Clamping keeps the maximum dose as 0xFFFFFFFF.
        scaled = np.clip(slice_2d / dose_grid_scaling, 0.0, uint32_max_value)
        pixel_frames[iz] = scaled.astype(np.uint32).T  # (H, W)

    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")

    study_instance_uid = study_instance_uid or generate_uid()
    frame_of_reference_uid = frame_of_reference_uid or generate_uid()
    series_instance_uid = series_instance_uid or generate_uid()
    sop_instance_uid = generate_uid()

    try:
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = RTDOSE_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # --- SOP common ---
        ds.SOPClassUID = RTDOSE_SOP_CLASS
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
        ds.Modality = "RTDOSE"
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = int(series_number)
        ds.SeriesDescription = series_description
        ds.SeriesDate = date_str
        ds.SeriesTime = time_str
        ds.Manufacturer = "DICOMator"
        ds.InstitutionName = "Virtual Hospital"
        ds.StationName = "Blender"

        # --- Frame of reference ---
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.PatientPosition = str(patient_position)

        # --- General image / multi-frame ---
        ds.InstanceNumber = "1"
        ds.ContentDate = date_str
        ds.ContentTime = time_str
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.NumberOfFrames = int(depth)

        # --- Image plane ---
        # ImagePositionPatient is the center of the first voxel (PS3.3
        # C.7.6.2.1.1), not the grid corner.  Offset by half a voxel so the
        # dose grid aligns with the co-exported CT/RT Structure Set.
        ds.ImagePositionPatient = [
            float(bbox_min_mm.x + 0.5 * vx_mm),
            float(bbox_min_mm.y + 0.5 * vy_mm),
            float(bbox_min_mm.z + 0.5 * vz_mm),
        ]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [float(vy_mm), float(vx_mm)]
        ds.SliceThickness = float(vz_mm)

        # GridFrameOffsetVector: Z offsets in mm for each frame relative to
        # ImagePositionPatient Z.  Frame 0 is at offset 0.
        ds.GridFrameOffsetVector = [float(iz * vz_mm) for iz in range(depth)]

        # --- Pixel dimensions ---
        ds.Rows = int(height)
        ds.Columns = int(width)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 32
        ds.BitsStored = 32
        ds.HighBit = 31
        ds.PixelRepresentation = 0  # unsigned

        # --- RT Dose specific ---
        ds.DoseUnits = "GY"
        ds.DoseType = str(dose_type or "PHYSICAL").upper()
        ds.DoseSummationType = str(dose_summation_type or "PLAN").upper()
        ds.DoseGridScaling = float(dose_grid_scaling)

        # Pixel data: all frames concatenated in slice order (C-contiguous).
        ds.PixelData = pixel_frames.tobytes()

        if phase_index is not None:
            filename = os.path.join(output_dir, f"Phase_{int(phase_index):03d}_RTDose.dcm")
        else:
            filename = os.path.join(output_dir, "RTDose.dcm")

        ds.save_as(filename)

    except Exception as exc:
        return {"error": f"Error saving RT Dose DICOM file: {exc}"}

    return {"success": f"Successfully exported RT Dose to {filename}"}


__all__ = ["export_rtdose_to_dicom"]
