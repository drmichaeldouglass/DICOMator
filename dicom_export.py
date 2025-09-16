"""DICOM export helpers used by the DICOMator add-on."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Optional, Sequence

import numpy as np
from mathutils import Vector

from .constants import (
    AIR_DENSITY,
    DEFAULT_DENSITY,
    MAX_HU_VALUE,
    MIN_HU_VALUE,
    Dataset,
    FileDataset,
    PYDICOM_AVAILABLE,
    generate_uid,
    pydicom,
)

SliceProgressCallback = Optional[Callable[[int, int], None]]


def export_voxel_grid_to_dicom(
    voxel_grid: np.ndarray,
    voxel_size: Sequence[float] | float,
    output_dir: str,
    bbox_min: Vector,
    *,
    patient_name: str = "Anonymous",
    patient_id: str = "12345678",
    patient_sex: str = "M",
    series_description: str = "CT Series from DICOMator",
    progress_callback: SliceProgressCallback = None,
    direct_hu: bool = False,
    patient_position: str = "HFS",
    study_instance_uid: Optional[str] = None,
    frame_of_reference_uid: Optional[str] = None,
    series_instance_uid: Optional[str] = None,
    series_number: int = 1,
    temporal_position_identifier: Optional[int] = None,
    number_of_temporal_positions: Optional[int] = None,
    phase_index: Optional[int] = None,
    temporal_position_index: Optional[int] = None,
) -> dict[str, str]:
    """Export ``voxel_grid`` to a folder of DICOM slices."""
    if not PYDICOM_AVAILABLE or pydicom is None or Dataset is None or FileDataset is None or generate_uid is None:
        return {'error': 'pydicom not available'}

    os.makedirs(output_dir, exist_ok=True)

    current_datetime = datetime.now()
    date_str = current_datetime.strftime('%Y%m%d')
    time_str = current_datetime.strftime('%H%M%S.%f')

    num_slices = voxel_grid.shape[2]
    if direct_hu:
        hu_grid = np.array(voxel_grid, dtype=np.int16, copy=False)
    else:
        hu_grid = np.where(voxel_grid > 0, DEFAULT_DENSITY, AIR_DENSITY).astype(np.int16, copy=False)
    hu_grid = np.clip(hu_grid, MIN_HU_VALUE, MAX_HU_VALUE).astype(np.int16, copy=False)

    if isinstance(voxel_size, Sequence) and len(voxel_size) == 3:
        vx_m, vy_m, vz_m = (float(component) for component in voxel_size)
    else:
        vx_m = vy_m = vz_m = float(voxel_size)
    vx_mm, vy_mm, vz_mm = vx_m * 1000.0, vy_m * 1000.0, vz_m * 1000.0

    bbox_min_mm = Vector((bbox_min.x * 1000.0, bbox_min.y * 1000.0, bbox_min.z * 1000.0))

    study_instance_uid = study_instance_uid or generate_uid()
    frame_of_reference_uid = frame_of_reference_uid or generate_uid()
    series_instance_uid = series_instance_uid or generate_uid()

    common_metadata = {
        'study_instance_uid': study_instance_uid,
        'frame_of_reference_uid': frame_of_reference_uid,
        'series_instance_uid': series_instance_uid,
        'series_number': series_number,
        'series_description': series_description,
        'patient_name': patient_name,
        'patient_id': patient_id,
        'patient_sex': patient_sex,
        'date_str': date_str,
        'time_str': time_str,
    }

    for index in range(num_slices):
        slice_data = hu_grid[:, :, index]
        try:
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

            dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
            dataset.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
            dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

            dataset.PatientName = common_metadata['patient_name']
            dataset.PatientID = common_metadata['patient_id']
            dataset.PatientBirthDate = ''
            dataset.PatientSex = common_metadata['patient_sex']
            dataset.PatientPosition = str(patient_position)

            dataset.StudyInstanceUID = common_metadata['study_instance_uid']
            dataset.FrameOfReferenceUID = common_metadata['frame_of_reference_uid']
            dataset.StudyID = '1'
            dataset.AccessionNumber = '1'
            dataset.StudyDate = common_metadata['date_str']
            dataset.StudyTime = common_metadata['time_str']
            dataset.ReferringPhysicianName = ''

            dataset.SeriesInstanceUID = common_metadata['series_instance_uid']
            dataset.SeriesNumber = common_metadata['series_number']
            dataset.SeriesDescription = common_metadata['series_description']
            dataset.SeriesDate = common_metadata['date_str']
            dataset.SeriesTime = common_metadata['time_str']

            if number_of_temporal_positions is not None:
                dataset.NumberOfTemporalPositions = int(number_of_temporal_positions)
            if temporal_position_index is not None:
                dataset.TemporalPositionIndex = int(temporal_position_index)
            if temporal_position_identifier is not None:
                dataset.TemporalPositionIdentifier = int(temporal_position_identifier)

            dataset.Modality = 'CT'
            dataset.Manufacturer = 'DICOMator'
            dataset.InstitutionName = 'Virtual Hospital'
            dataset.StationName = 'Blender'

            dataset.InstanceNumber = index + 1
            dataset.AcquisitionNumber = int(phase_index or 1)
            dataset.ContentDate = common_metadata['date_str']
            dataset.ContentTime = common_metadata['time_str']
            dataset.AcquisitionDate = common_metadata['date_str']
            dataset.AcquisitionTime = common_metadata['time_str']

            dataset.ImagePositionPatient = [
                float(bbox_min_mm.x),
                float(bbox_min_mm.y),
                float(bbox_min_mm.z + (index * vz_mm)),
            ]
            dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            dataset.SliceLocation = float(bbox_min_mm.z + (index * vz_mm))
            dataset.SliceThickness = float(vz_mm)
            dataset.SpacingBetweenSlices = float(vz_mm)

            pixel_array = slice_data.T.astype(np.int16, copy=False)
            rows, cols = pixel_array.shape
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = 'MONOCHROME2'
            dataset.Rows = int(rows)
            dataset.Columns = int(cols)
            dataset.PixelSpacing = [float(vy_mm), float(vx_mm)]
            dataset.BitsAllocated = 16
            dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 1
            dataset.RescaleIntercept = 0.0
            dataset.RescaleSlope = 1.0
            dataset.WindowCenter = 40
            dataset.WindowWidth = 400
            dataset.PixelData = pixel_array.tobytes()

            if phase_index is not None:
                filename = os.path.join(output_dir, f"Phase_{int(phase_index):03d}_CT_Slice_{index + 1:04d}.dcm")
            else:
                filename = os.path.join(output_dir, f"CT_Slice_{index + 1:04d}.dcm")

            dataset.is_little_endian = True
            dataset.is_implicit_VR = False
            dataset.save_as(filename)

            if progress_callback:
                progress_callback(index + 1, num_slices)
        except Exception as exc:  # pragma: no cover - Blender runtime feedback
            print(f"Error saving DICOM file for slice {index + 1}: {exc}")
            return {'error': f"Error saving DICOM file for slice {index + 1}: {exc}"}

    return {'success': f"Successfully exported {num_slices} DICOM slices to {output_dir}"}


__all__ = ["export_voxel_grid_to_dicom"]
