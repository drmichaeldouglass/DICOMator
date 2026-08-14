"""Tests for atomic export-directory handling and memory preflight."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import load_module

constants = load_module("constants")
operators = load_module("operators")


def test_atomic_output_commit_replaces_empty_destination(tmp_path):
    final = tmp_path / "export"
    final.mkdir()
    final_path, staging = operators._prepare_atomic_output_directory(str(final))
    staged_file = Path(staging) / "CT_Slice_0001.dcm"
    staged_file.write_bytes(b"complete")

    operators._finalize_atomic_output_directory(staging, final_path, commit=True)

    assert (final / "CT_Slice_0001.dcm").read_bytes() == b"complete"
    assert not Path(staging).exists()


def test_atomic_output_cancel_removes_partial_staging(tmp_path):
    final = tmp_path / "export"
    final_path, staging = operators._prepare_atomic_output_directory(str(final))
    Path(staging, "partial.dcm").write_bytes(b"partial")

    operators._finalize_atomic_output_directory(staging, final_path, commit=False)

    assert not Path(staging).exists()
    assert not final.exists()


def test_atomic_output_rejects_nonempty_destination(tmp_path):
    final = tmp_path / "export"
    final.mkdir()
    (final / "old-study.dcm").write_bytes(b"old")

    with pytest.raises(ValueError, match="not empty"):
        operators._prepare_atomic_output_directory(str(final))

    assert (final / "old-study.dcm").read_bytes() == b"old"


def test_memory_estimate_accounts_for_rtdose_temporaries():
    total_voxels = 100_000_000
    estimated = constants.estimate_peak_memory_bytes(
        total_voxels,
        export_image_series=False,
        export_drr=False,
        export_rtdose=True,
        artifacts_enabled=False,
        gibbs_enabled=False,
    )
    assert estimated == total_voxels * 16


def test_property_snapshot_is_independent_of_later_edits():
    props = SimpleNamespace(patient_name="Before", artifact_seed=7)
    snapshot = operators._snapshot_properties(props)
    props.patient_name = "After"
    props.artifact_seed = 8
    assert snapshot.patient_name == "Before"
    assert snapshot.artifact_seed == 7
