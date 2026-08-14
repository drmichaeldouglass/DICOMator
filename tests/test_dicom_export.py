"""Round-trip tests for the CT/MR slice writer and the DRR writer."""
from __future__ import annotations

import numpy as np
import pytest
from mathutils import Vector

from conftest import load_module

constants = load_module("constants")
dicom_export = load_module("dicom_export")

pydicom = pytest.importorskip("pydicom")


@pytest.fixture(scope="module", autouse=True)
def _require_pydicom():
    assert constants.ensure_pydicom_available()


def _grid(shape=(8, 6, 4)) -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.integers(-1000, 2000, size=shape, dtype=np.int16)


VOXEL_SIZE_M = (0.002, 0.003, 0.004)  # 2 x 3 x 4 mm
BBOX_MIN = Vector((0.01, -0.02, 0.05))  # metres


def _export(tmp_path, grid, **kwargs):
    result = dicom_export.export_voxel_grid_to_dicom(
        grid,
        VOXEL_SIZE_M,
        str(tmp_path),
        BBOX_MIN,
        direct_hu=True,
        **kwargs,
    )
    assert "success" in result, result
    files = sorted(tmp_path.glob("*.dcm"))
    return result, [pydicom.dcmread(str(path)) for path in files]


def test_ct_series_geometry_and_pixels(tmp_path):
    grid = _grid()
    result, datasets = _export(tmp_path, grid)

    assert len(datasets) == grid.shape[2]
    assert len(result["sop_instance_uids"]) == grid.shape[2]

    study_uids = {ds.StudyInstanceUID for ds in datasets}
    series_uids = {ds.SeriesInstanceUID for ds in datasets}
    frame_uids = {ds.FrameOfReferenceUID for ds in datasets}
    assert len(study_uids) == len(series_uids) == len(frame_uids) == 1

    vx_mm, vy_mm, vz_mm = (v * 1000.0 for v in VOXEL_SIZE_M)
    for index, ds in enumerate(datasets):
        assert int(ds.InstanceNumber) == index + 1
        assert ds.Modality == "CT"
        assert list(ds.ImageType) == ["DERIVED", "PRIMARY", "AXIAL"]
        assert ds.SyntheticData == "YES"
        assert ds.SpecificCharacterSet == "ISO_IR 192"
        assert "Blender geometry" in ds.DerivationDescription
        assert "PositionReferenceIndicator" in ds
        assert ds.RescaleType == "HU"
        # ImagePositionPatient is the centre of the first voxel.
        np.testing.assert_allclose(
            [float(v) for v in ds.ImagePositionPatient],
            [
                BBOX_MIN.x * 1000.0 + 0.5 * vx_mm,
                BBOX_MIN.y * 1000.0 + 0.5 * vy_mm,
                BBOX_MIN.z * 1000.0 + (index + 0.5) * vz_mm,
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose([float(v) for v in ds.PixelSpacing], [vy_mm, vx_mm], atol=1e-9)
        assert int(ds.Rows) == grid.shape[1]
        assert int(ds.Columns) == grid.shape[0]
        np.testing.assert_array_equal(ds.pixel_array, grid[:, :, index].T)


def test_shared_uids_are_respected(tmp_path):
    generate_uid = constants.generate_uid
    study_uid = generate_uid()
    frame_uid = generate_uid()
    _, datasets = _export(
        tmp_path,
        _grid(),
        study_instance_uid=study_uid,
        frame_of_reference_uid=frame_uid,
    )
    assert all(ds.StudyInstanceUID == study_uid for ds in datasets)
    assert all(ds.FrameOfReferenceUID == frame_uid for ds in datasets)


def test_mr_series_has_mr_module(tmp_path):
    _, datasets = _export(tmp_path, _grid(), dicom_modality="MR")
    for ds in datasets:
        assert ds.Modality == "MR"
        assert list(ds.ImageType) == ["DERIVED", "PRIMARY", "OTHER"]
        assert ds.ScanningSequence == "SE"
        assert float(ds.RepetitionTime) > 0.0
        assert float(ds.EchoTime) > 0.0
        assert "RescaleSlope" not in ds


def test_mr_tags_follow_weighting_preset(tmp_path):
    t1_dir = tmp_path / "t1"
    t2_dir = tmp_path / "t2"
    _, t1_datasets = _export(t1_dir, _grid(), dicom_modality="MR", mr_weighting=constants.MODALITY_MRI_T1)
    _, t2_datasets = _export(t2_dir, _grid(), dicom_modality="MR", mr_weighting=constants.MODALITY_MRI_T2)

    for ds in t1_datasets:
        assert str(ds.RepetitionTime) == "500"
        assert str(ds.EchoTime) == "15"
        assert str(ds.EchoTrainLength) == "1"
        assert str(ds.SequenceVariant) == "NONE"
    for ds in t2_datasets:
        assert str(ds.RepetitionTime) == "4000"
        assert str(ds.EchoTime) == "100"
        assert str(ds.EchoTrainLength) == "16"
        assert str(ds.SequenceVariant) == "SK"


def test_study_metadata_and_shared_timestamp(tmp_path):
    from datetime import datetime

    stamp = datetime(2026, 7, 8, 12, 34, 56, 789000)
    _, datasets = _export(
        tmp_path,
        _grid(),
        study_id="PHANTOM-01",
        accession_number="ACC-42",
        patient_birth_date="1980-02-01",
        study_datetime=stamp,
    )
    for ds in datasets:
        assert str(ds.StudyID) == "PHANTOM-01"
        assert str(ds.AccessionNumber) == "ACC-42"
        assert str(ds.PatientBirthDate) == "19800201"
        assert str(ds.StudyDate) == "20260708"
        assert str(ds.StudyTime) == "123456.789000"
        assert str(ds.ContentDate) == "20260708"
        assert str(ds.ContentTime) == "123456.789000"

    drr_result = dicom_export.export_projection_to_dicom(
        np.zeros((4, 4), dtype=np.uint16),
        str(tmp_path),
        filename="DRR_meta.dcm",
        study_id="PHANTOM-01",
        accession_number="ACC-42",
        patient_birth_date="19800201",
        study_datetime=stamp,
    )
    assert "success" in drr_result, drr_result
    drr = pydicom.dcmread(str(tmp_path / "DRR_meta.dcm"))
    assert str(drr.StudyDate) == "20260708"
    assert str(drr.StudyTime) == "123456.789000"
    assert str(drr.StudyID) == "PHANTOM-01"
    assert str(drr.PatientBirthDate) == "19800201"


def test_invalid_birth_date_written_empty(tmp_path):
    _, datasets = _export(tmp_path, _grid(), patient_birth_date="Feb 1980")
    for ds in datasets:
        assert str(ds.PatientBirthDate) == ""


def test_impossible_birth_date_written_empty(tmp_path):
    _, datasets = _export(tmp_path, _grid(), patient_birth_date="2023-02-31")
    assert all(str(ds.PatientBirthDate) == "" for ds in datasets)


def test_text_is_utf8_sanitized_and_byte_limited(tmp_path):
    _, datasets = _export(
        tmp_path,
        _grid(),
        patient_name="Müller\\Injected\x7f" + "患者" * 30,
        patient_id="患者" * 30,
        series_description="Séries\\Second value",
    )
    ds = datasets[0]
    assert "\\" not in str(ds.PatientName)
    assert "\x7f" not in str(ds.PatientName)
    assert "\\" not in str(ds.SeriesDescription)
    assert len(str(ds.PatientName).encode("utf-8")) <= 64
    assert len(str(ds.PatientID).encode("utf-8")) <= 64
    assert ds.SpecificCharacterSet == "ISO_IR 192"


def test_long_study_id_truncated_to_sh_limit(tmp_path):
    _, datasets = _export(tmp_path, _grid(), study_id="X" * 40)
    for ds in datasets:
        assert str(ds.StudyID) == "X" * 16


def test_normalize_and_truncate_helpers():
    assert constants.normalize_dicom_date("1980-02-01") == "19800201"
    assert constants.normalize_dicom_date("19800201") == "19800201"
    assert constants.normalize_dicom_date("1980") == ""
    assert constants.normalize_dicom_date("20230231") == ""
    assert constants.normalize_dicom_date(None) == ""
    assert constants.truncate_sh("A" * 20) == "A" * 16
    assert constants.truncate_sh("", "1") == "1"
    assert constants.truncate_sh("  ", "1") == "1"
    assert constants.truncate_sh(None, "1") == "1"
    assert len(constants.truncate_sh("患者" * 20).encode("utf-8")) <= 16


def test_direct_hu_clips_before_int16_cast(tmp_path):
    grid = np.array([40000.0, -40000.0, 5000.0], dtype=np.float32).reshape(1, 1, 3)
    _, datasets = _export(tmp_path, grid)
    assert [int(ds.pixel_array[0, 0]) for ds in datasets] == [3071, -1024, 3071]


@pytest.mark.parametrize(
    "grid,voxel_size,error_text",
    [
        (np.empty((0, 2, 2)), VOXEL_SIZE_M, "must not be empty"),
        (np.zeros((2, 2)), VOXEL_SIZE_M, "must be a 3D array"),
        (np.full((2, 2, 2), np.nan), VOXEL_SIZE_M, "NaN or infinite"),
        (np.zeros((2, 2, 2)), (0.002, 0.0, 0.004), "greater than zero"),
    ],
)
def test_image_writer_rejects_invalid_inputs(tmp_path, grid, voxel_size, error_text):
    result = dicom_export.export_voxel_grid_to_dicom(
        grid,
        voxel_size,
        str(tmp_path),
        BBOX_MIN,
        direct_hu=True,
    )
    assert error_text in result["error"]
    assert not list(tmp_path.glob("*.dcm"))


def test_image_writer_rejects_nonfinite_origin(tmp_path):
    result = dicom_export.export_voxel_grid_to_dicom(
        _grid(),
        VOXEL_SIZE_M,
        str(tmp_path),
        Vector((0.0, float("inf"), 0.0)),
        direct_hu=True,
    )
    assert "bbox_min components must be finite" in result["error"]
    assert not list(tmp_path.glob("*.dcm"))


def test_drr_projection_roundtrip(tmp_path):
    image = (np.arange(6 * 8, dtype=np.uint16) * 900).reshape(6, 8)
    result = dicom_export.export_projection_to_dicom(
        image,
        str(tmp_path),
        filename="DRR_test.dcm",
        pixel_spacing_mm=(1.5, 2.5),
    )
    assert "success" in result, result
    ds = pydicom.dcmread(str(tmp_path / "DRR_test.dcm"))
    assert ds.Modality == "OT"
    assert ds.SyntheticData == "YES"
    assert ds.SpecificCharacterSet == "ISO_IR 192"
    assert "PositionReferenceIndicator" in ds
    assert int(ds.Rows) == 6
    assert int(ds.Columns) == 8
    np.testing.assert_allclose([float(v) for v in ds.PixelSpacing], [1.5, 2.5])
    np.testing.assert_array_equal(ds.pixel_array, image)
