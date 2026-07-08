"""Tests for the Blender-free RT Structure Set dataset builder."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import load_module

constants = load_module("constants")
rtstruct_export = load_module("rtstruct_export")

pydicom = pytest.importorskip("pydicom")


@pytest.fixture(scope="module", autouse=True)
def _require_pydicom():
    assert constants.ensure_pydicom_available()


SQUARE_M = [(0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.0, 0.05)]
VZ_M = 0.003
BBOX_MIN_Z_M = -0.01


def _loop_at(z_m: float):
    return [(x, y, z_m) for x, y in SQUARE_M]


def _z_for_slice(index: int) -> float:
    return BBOX_MIN_Z_M + (index + 0.5) * VZ_M


def _build(**overrides):
    generate_uid = constants.generate_uid
    ct_uids = [generate_uid() for _ in range(4)]
    z0 = _z_for_slice(1)
    z1 = _z_for_slice(2)
    roi_defs = [
        (
            "A very long structure name exceeding limits",
            (255, 0, 0),
            "GTV",
            {z0: [_loop_at(z0)], z1: [_loop_at(z1)]},
        ),
        ("Cord", (0, 0, 255), "OAR", {z0: [_loop_at(z0)]}),
    ]
    kwargs = dict(
        study_instance_uid=generate_uid(),
        frame_of_reference_uid=generate_uid(),
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        date_str="20260708",
        time_str="120000.000000",
        series_description="Structures from a long series description",
        bbox_min_z_m=BBOX_MIN_Z_M,
        vz_m=VZ_M,
        referenced_ct_series_instance_uid=generate_uid(),
        referenced_ct_sop_class_uid=str(pydicom.uid.CTImageStorage),
        referenced_ct_sop_instance_uids=ct_uids,
    )
    kwargs.update(overrides)
    return rtstruct_export.build_rtstruct_dataset(roi_defs, **kwargs), kwargs, ct_uids


def test_sequences_and_cross_references():
    ds, kwargs, _ = _build()

    assert ds.SOPClassUID == constants.RTSTRUCT_SOP_CLASS
    assert ds.Modality == "RTSTRUCT"
    assert ds.StudyInstanceUID == kwargs["study_instance_uid"]
    assert (
        ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
        == kwargs["frame_of_reference_uid"]
    )

    assert len(ds.StructureSetROISequence) == 2
    assert len(ds.ROIContourSequence) == 2
    assert len(ds.RTROIObservationsSequence) == 2

    for index, (roi, contour, obs) in enumerate(
        zip(ds.StructureSetROISequence, ds.ROIContourSequence, ds.RTROIObservationsSequence),
        start=1,
    ):
        assert int(roi.ROINumber) == index
        assert int(contour.ReferencedROINumber) == index
        assert int(obs.ReferencedROINumber) == index
        assert roi.ReferencedFrameOfReferenceUID == kwargs["frame_of_reference_uid"]

    assert ds.RTROIObservationsSequence[0].RTROIInterpretedType == "GTV"
    assert ds.RTROIObservationsSequence[1].RTROIInterpretedType == "OAR"


def test_label_truncation():
    ds, _, _ = _build()
    assert len(str(ds.StructureSetLabel)) <= 16
    assert len(str(ds.RTROIObservationsSequence[0].ROIObservationLabel)) <= 16
    # ROIName is VR LO (64 chars) and keeps the full name here.
    assert ds.StructureSetROISequence[0].ROIName == "A very long structure name exceeding limits"


def test_contour_geometry_and_mm_conversion():
    ds, _, _ = _build()
    contour = ds.ROIContourSequence[0].ContourSequence[0]
    assert contour.ContourGeometricType == "CLOSED_PLANAR"
    assert int(contour.NumberOfContourPoints) == 4

    data = [float(v) for v in contour.ContourData]
    assert len(data) == 12
    z0_mm = _z_for_slice(1) * 1000.0
    expected = []
    for x_m, y_m in SQUARE_M:
        expected.extend([x_m * 1000.0, y_m * 1000.0, z0_mm])
    np.testing.assert_allclose(data, expected, atol=1e-4)


def test_contour_image_sequence_maps_z_to_ct_slice():
    ds, _, ct_uids = _build()
    contours = ds.ROIContourSequence[0].ContourSequence
    # First ROI has contours on slice indices 1 and 2.
    referenced = [item.ContourImageSequence[0].ReferencedSOPInstanceUID for item in contours]
    assert referenced == [ct_uids[1], ct_uids[2]]


def test_no_ct_reference_omits_study_sequence():
    ds, _, _ = _build(
        referenced_ct_series_instance_uid=None,
        referenced_ct_sop_class_uid=None,
        referenced_ct_sop_instance_uids=None,
    )
    ref_frame = ds.ReferencedFrameOfReferenceSequence[0]
    assert "RTReferencedStudySequence" not in ref_frame
    for item in ds.ROIContourSequence[0].ContourSequence:
        assert "ContourImageSequence" not in item


def test_display_colour_round_trip():
    ds, _, _ = _build()
    assert [int(v) for v in ds.ROIContourSequence[0].ROIDisplayColor] == [255, 0, 0]
    assert [int(v) for v in ds.ROIContourSequence[1].ROIDisplayColor] == [0, 0, 255]
