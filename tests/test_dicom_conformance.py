"""DICOM value-representation conformance for every writer.

DICOM caps a Decimal String (DS) value at 16 characters (PS3.5 Table 6.2-1),
but Python's shortest round-trip float repr routinely needs more, e.g.
``-122.74999999999999`` for a voxel centre. These tests pin the formatting
helper and then check each writer's output with voxel sizes and a bounding box
chosen so the raw floats would overflow the limit.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
from mathutils import Vector

from conftest import load_module

constants = load_module("constants")
dicom_export = load_module("dicom_export")
rtdose_export = load_module("rtdose_export")
rtstruct_export = load_module("rtstruct_export")

pydicom = pytest.importorskip("pydicom")

# 1.3 mm / 2.7 mm spacing and an off-grid origin: every derived coordinate
# lands on a float whose repr needs 18-19 characters.
VOXEL_SIZE_M = (0.0013, 0.0013, 0.0027)
BBOX_MIN = Vector((-0.1234, 0.0567, -0.4321))
DS_MAX_LEN = 16


@pytest.fixture(scope="module", autouse=True)
def _require_pydicom():
    assert constants.ensure_pydicom_available()


def _overlong_ds_values(dataset) -> list[tuple[str, str]]:
    """Return ``(keyword, value)`` for every DS value longer than 16 bytes."""

    found: list[tuple[str, str]] = []

    def walk(item, prefix=""):
        for element in item:
            if element.VR == "SQ":
                for index, sub_item in enumerate(element.value or []):
                    walk(sub_item, f"{prefix}{element.keyword}[{index}].")
                continue
            if element.VR != "DS":
                continue
            values = element.value
            if not isinstance(values, (list, pydicom.multival.MultiValue)):
                values = [values]
            for value in values:
                text = str(value)
                if len(text.encode("utf-8")) > DS_MAX_LEN:
                    found.append((prefix + (element.keyword or str(element.tag)), text))

    walk(dataset)
    return found


def _assert_all_ds_conformant(directory: str) -> int:
    files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    assert files, "expected at least one exported DICOM file"
    for path in files:
        overlong = _overlong_ds_values(pydicom.dcmread(path))
        assert not overlong, f"{os.path.basename(path)} has non-conformant DS values: {overlong}"
    return len(files)


# ---------------------------------------------------------------------------
# format_ds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1.0,
        0.5,
        -0.5,
        1.3000000000000003,
        -122.74999999999999,
        0.9999999403953552,
        1.0 / 3.0,
        4.656612874161595e-10,
        -4.656612874161595e-10,
        1e-200,
        -1e-200,
        1.5e13,
        -1.234e12,
        1e15,
        -1e15,
        1e-5,
        12345.6789,
        # Rounding to the remaining precision can carry into an extra integer
        # digit, e.g. 9.999999999999998 -> '10.00000000000000' (17 characters).
        9.999999999999998,
        -9.999999999999998,
        99.99999999999999,
        999.9999999999999,
        0.9999999999999999,
        99999999999999.98,
        -99999999999999.98,
    ],
)
def test_format_ds_fits_the_limit_and_round_trips(value):
    text = constants.format_ds(value)
    assert len(text.encode("utf-8")) <= DS_MAX_LEN, text
    # DS permits only digits, sign, decimal point, and the exponent marker.
    assert set(text) <= set("0123456789+-.eE"), text
    # The string must still parse back to (very nearly) the same number.
    assert float(text) == pytest.approx(value, rel=1e-8, abs=1e-12)


def test_format_ds_keeps_short_reprs_verbatim():
    assert constants.format_ds(2.0) == "2.0"
    assert constants.format_ds(12345.6789) == "12345.6789"


def test_format_ds_rejects_non_finite():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            constants.format_ds(value)


def test_format_ds_sequence_formats_every_component():
    formatted = constants.format_ds_sequence((1.3000000000000003, -122.74999999999999))
    assert formatted == ["1.30000000000000", "-122.75000000000"]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_image_series_ds_values_are_conformant(tmp_path):
    grid = np.zeros((3, 4, 5), dtype=np.int16)
    result = dicom_export.export_voxel_grid_to_dicom(
        grid, VOXEL_SIZE_M, str(tmp_path), BBOX_MIN, direct_hu=True
    )
    assert "success" in result, result
    assert _assert_all_ds_conformant(str(tmp_path)) == 5


def test_image_series_geometry_survives_ds_formatting(tmp_path):
    grid = np.zeros((3, 4, 5), dtype=np.int16)
    dicom_export.export_voxel_grid_to_dicom(
        grid, VOXEL_SIZE_M, str(tmp_path), BBOX_MIN, direct_hu=True
    )
    vx_mm, vy_mm, vz_mm = (size * 1000.0 for size in VOXEL_SIZE_M)
    for index in range(5):
        dataset = pydicom.dcmread(str(tmp_path / f"CT_Slice_{index + 1:04d}.dcm"))
        position = [float(value) for value in dataset.ImagePositionPatient]
        assert position[0] == pytest.approx(BBOX_MIN.x * 1000.0 + 0.5 * vx_mm, abs=1e-6)
        assert position[1] == pytest.approx(BBOX_MIN.y * 1000.0 + 0.5 * vy_mm, abs=1e-6)
        assert position[2] == pytest.approx(
            BBOX_MIN.z * 1000.0 + (index + 0.5) * vz_mm, abs=1e-6
        )
        assert float(dataset.SliceLocation) == pytest.approx(position[2], abs=1e-6)
        assert [float(value) for value in dataset.PixelSpacing] == pytest.approx(
            [vy_mm, vx_mm], abs=1e-9
        )
        assert float(dataset.SliceThickness) == pytest.approx(vz_mm, abs=1e-9)


def test_drr_ds_values_are_conformant(tmp_path):
    # Direction cosines come out of single-precision mathutils vectors, so
    # components such as 0.9999999403953552 reach the writer.
    result = dicom_export.export_projection_to_dicom(
        np.zeros((4, 5), dtype=np.uint16),
        str(tmp_path),
        pixel_spacing_mm=(0.3671875000000001, 0.3671875000000001),
        image_position_patient=(-122.74999999999999, 56.7, -432.09999999999997),
        image_orientation_patient=(
            0.9999999403953552,
            -1.1920928955078125e-07,
            0.0,
            0.0,
            -0.9999999403953552,
            1.1920928955078125e-07,
        ),
    )
    assert "success" in result, result
    assert _assert_all_ds_conformant(str(tmp_path)) == 1


def test_rtdose_ds_values_are_conformant(tmp_path):
    grid = np.full((3, 4, 5), 2.0, dtype=np.float32)
    result = rtdose_export.export_rtdose_to_dicom(
        grid, VOXEL_SIZE_M, BBOX_MIN, str(tmp_path)
    )
    assert "success" in result, result
    # RT Dose and its companion RT Plan.
    assert _assert_all_ds_conformant(str(tmp_path)) == 2


def test_rtstruct_ds_values_are_conformant(tmp_path):
    vz_m = VOXEL_SIZE_M[2]
    z_m = BBOX_MIN.z + 0.5 * vz_m
    roi_defs = [
        (
            "Body",
            (255, 0, 0),
            "EXTERNAL",
            {z_m: [[(-0.1234, 0.0567, z_m), (0.1, 0.2, z_m), (0.05, 0.3, z_m)]]},
        )
    ]
    dataset = rtstruct_export.build_rtstruct_dataset(
        roi_defs,
        study_instance_uid="1.2.3",
        frame_of_reference_uid="1.2.4",
        series_instance_uid="1.2.5",
        sop_instance_uid="1.2.6",
        date_str="20260101",
        time_str="120000.000000",
        bbox_min_z_m=float(BBOX_MIN.z),
        vz_m=vz_m,
    )
    path = tmp_path / "RTStruct.dcm"
    dataset.save_as(str(path), enforce_file_format=True)
    assert _assert_all_ds_conformant(str(tmp_path)) == 1


# ---------------------------------------------------------------------------
# DoseGridScaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("peak_dose", [0.007, 2.0, 60.0, 123.456])
def test_dose_grid_scaling_is_conformant_and_preserves_dose(tmp_path, peak_dose):
    rng = np.random.default_rng(4)
    grid = rng.uniform(0.0, peak_dose, size=(6, 5, 4)).astype(np.float32)
    grid[0, 0, 0] = peak_dose

    result = rtdose_export.export_rtdose_to_dicom(
        grid, VOXEL_SIZE_M, BBOX_MIN, str(tmp_path)
    )
    assert "success" in result, result

    dataset = pydicom.dcmread(str(tmp_path / "RTDose.dcm"))
    scaling_text = str(dataset["DoseGridScaling"].value)
    # The naive max_dose / uint32_max factor needs ~21 characters as a DS.
    assert len(scaling_text.encode("utf-8")) <= DS_MAX_LEN, scaling_text

    counts = dataset.pixel_array
    assert int(counts.max()) <= np.iinfo(np.uint32).max

    # The dose must be recoverable from the factor that was actually written,
    # not from the un-representable ideal factor.
    reconstructed = counts.astype(np.float64).transpose(2, 1, 0) * float(scaling_text)
    assert float(reconstructed.max()) == pytest.approx(peak_dose, rel=1e-6)
    np.testing.assert_allclose(reconstructed, grid, rtol=1e-5, atol=peak_dose * 1e-6)
