"""Round-trip tests for the RT Dose writer and its companion RT Plan."""
from __future__ import annotations

import numpy as np
import pytest
from mathutils import Vector

from conftest import load_module

constants = load_module("constants")
rtdose_export = load_module("rtdose_export")

pydicom = pytest.importorskip("pydicom")

VOXEL_SIZE_M = (0.002, 0.0025, 0.003)
BBOX_MIN = Vector((0.0, 0.01, -0.02))


@pytest.fixture(scope="module", autouse=True)
def _require_pydicom():
    assert constants.ensure_pydicom_available()


def _dose_grid(shape=(6, 5, 4), max_dose=60.0) -> np.ndarray:
    rng = np.random.default_rng(8)
    grid = rng.uniform(0.0, max_dose, size=shape).astype(np.float32)
    grid[0, 0, 0] = max_dose
    return grid


def test_rtdose_roundtrip(tmp_path):
    grid = _dose_grid()
    result = rtdose_export.export_rtdose_to_dicom(grid, VOXEL_SIZE_M, BBOX_MIN, str(tmp_path))
    assert "success" in result, result

    ds = pydicom.dcmread(str(tmp_path / "RTDose.dcm"))
    width, height, depth = grid.shape
    assert ds.Modality == "RTDOSE"
    assert int(ds.NumberOfFrames) == depth
    assert int(ds.Rows) == height
    assert int(ds.Columns) == width
    assert ds.DoseUnits == "GY"

    scaling = float(ds.DoseGridScaling)
    recon = ds.pixel_array.astype(np.float64) * scaling  # (frames, rows, cols)
    # Precision is bounded by float32 rounding of the scaled values, not by
    # the uint32 quantization itself.
    assert abs(float(recon.max()) - 60.0) / 60.0 < 1e-6
    np.testing.assert_allclose(recon.transpose(2, 1, 0), grid, rtol=1e-5, atol=1e-4)

    vz_mm = VOXEL_SIZE_M[2] * 1000.0
    np.testing.assert_allclose(
        [float(v) for v in ds.GridFrameOffsetVector],
        [iz * vz_mm for iz in range(depth)],
        atol=1e-9,
    )


def test_rtdose_zero_grid_uses_unit_scaling(tmp_path):
    grid = np.zeros((4, 4, 3), dtype=np.float32)
    result = rtdose_export.export_rtdose_to_dicom(grid, 0.002, BBOX_MIN, str(tmp_path))
    assert "success" in result, result
    ds = pydicom.dcmread(str(tmp_path / "RTDose.dcm"))
    assert float(ds.DoseGridScaling) == 1.0
    assert not np.any(ds.pixel_array)


def test_rtdose_references_companion_plan(tmp_path):
    result = rtdose_export.export_rtdose_to_dicom(_dose_grid(), VOXEL_SIZE_M, BBOX_MIN, str(tmp_path))
    assert "success" in result, result

    plan_path = tmp_path / "RTPlan.dcm"
    assert plan_path.exists()
    plan = pydicom.dcmread(str(plan_path))
    dose = pydicom.dcmread(str(tmp_path / "RTDose.dcm"))

    assert plan.Modality == "RTPLAN"
    ref = dose.ReferencedRTPlanSequence[0]
    assert ref.ReferencedSOPInstanceUID == plan.SOPInstanceUID
    assert plan.StudyInstanceUID == dose.StudyInstanceUID
    assert plan.FrameOfReferenceUID == dose.FrameOfReferenceUID


def test_rtdose_shares_provided_study_uid(tmp_path):
    study_uid = constants.generate_uid()
    result = rtdose_export.export_rtdose_to_dicom(
        _dose_grid(), VOXEL_SIZE_M, BBOX_MIN, str(tmp_path), study_instance_uid=study_uid
    )
    assert "success" in result, result
    ds = pydicom.dcmread(str(tmp_path / "RTDose.dcm"))
    assert ds.StudyInstanceUID == study_uid
