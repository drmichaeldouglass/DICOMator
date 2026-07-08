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
        assert ds.ScanningSequence == "SE"
        assert float(ds.RepetitionTime) > 0.0
        assert float(ds.EchoTime) > 0.0
        assert "RescaleSlope" not in ds


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
    assert int(ds.Rows) == 6
    assert int(ds.Columns) == 8
    np.testing.assert_allclose([float(v) for v in ds.PixelSpacing], [1.5, 2.5])
    np.testing.assert_array_equal(ds.pixel_array, image)
