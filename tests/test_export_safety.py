"""Tests for atomic export-directory handling and memory preflight."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT, load_module

constants = load_module("constants")
operators = load_module("operators")
panels = load_module("panels")


def _props(**overrides) -> SimpleNamespace:
    """Return a property stand-in with every export/artifact flag defined."""

    values = {
        "export_image_series": True,
        "export_drr": False,
        "export_rtdose": False,
        "lateral_resolution_mm": 2.0,
        "axial_resolution_mm": 2.0,
    }
    values.update({flag: False for flag in constants.ARTIFACT_FLAGS})
    values.update(overrides)
    return SimpleNamespace(**values)


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


# ---------------------------------------------------------------------------
# Grid guardrails
# ---------------------------------------------------------------------------


def test_grid_within_every_limit_is_allowed():
    assert not constants.grid_limits_exceeded(100, 100, 100, 1024)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_grid_limits_catch_each_oversized_dimension(axis):
    dimensions = [10, 10, 10]
    dimensions[axis] = constants.MAX_GRID_DIMENSION + 1
    assert constants.grid_limits_exceeded(*dimensions, 0)


def test_grid_limits_catch_total_voxel_count():
    side = 1000  # 1e9 voxels, but no single dimension is oversized
    assert side <= constants.MAX_GRID_DIMENSION
    assert constants.grid_limits_exceeded(side, side, side, 0)


def test_grid_limits_catch_memory_only_overruns():
    """The case the export button used to miss.

    440^3 stays under both the per-dimension and total voxel caps, yet with
    artifacts enabled the estimated peak allocation passes 2 GiB, so the
    operator aborts. The UI must reach the same verdict.
    """

    side = 440
    total_voxels = side**3
    assert side <= constants.MAX_GRID_DIMENSION
    assert total_voxels <= constants.MAX_TOTAL_VOXELS

    props = _props(enable_noise=True)
    memory_bytes = constants.estimate_peak_memory_bytes_for_props(total_voxels, props)
    assert memory_bytes > constants.MAX_ESTIMATED_MEMORY_BYTES
    assert constants.grid_limits_exceeded(side, side, side, memory_bytes)

    # Without artifacts the same grid fits, so the limit is genuinely the
    # memory estimate rather than the voxel count.
    lean_bytes = constants.estimate_peak_memory_bytes_for_props(total_voxels, _props())
    assert not constants.grid_limits_exceeded(side, side, side, lean_bytes)


def test_memory_estimate_from_props_matches_explicit_arguments():
    props = _props(export_drr=True, enable_gibbs_ringing=True)
    assert constants.estimate_peak_memory_bytes_for_props(1000, props) == (
        constants.estimate_peak_memory_bytes(
            1000,
            export_image_series=True,
            export_drr=True,
            export_rtdose=False,
            artifacts_enabled=True,
            gibbs_enabled=True,
        )
    )


def test_artifact_flags_cover_every_enable_toggle():
    """A new artifact toggle must reach the shared list, not just the UI."""

    source = (REPO_ROOT / "properties.py").read_text()
    declared = {
        line.split(":")[0].strip()
        for line in source.splitlines()
        if line.strip().startswith("enable_")
    }
    assert declared, "expected enable_* properties in properties.py"
    assert declared == set(constants.ARTIFACT_FLAGS)


@pytest.mark.parametrize(
    "dimensions_m",
    [(0.2, 0.3, 0.4), (0.05, 0.05, 0.05), (1.0, 0.75, 0.5), (0.123, 0.456, 0.789)],
)
def test_panel_estimate_matches_the_operator_grid(dimensions_m):
    """The panel's preview must predict the grid the export actually builds.

    Both apply one voxel of padding on every side; if they drift the UI will
    green-light a selection the operator then rejects (or vice versa).
    """

    props = _props(lateral_resolution_mm=1.3, axial_resolution_mm=2.7)
    voxel_size_m = (0.0013, 0.0013, 0.0027)

    est_width, est_height, est_depth, total_voxels = panels._grid_estimate(
        dimensions_m, props
    )
    assert total_voxels == est_width * est_height * est_depth

    width, height, depth = dimensions_m
    bounds = (0.0, width, 0.0, height, 0.0, depth)
    padded = operators._pad_bounds(bounds, voxel_size_m, 1)
    assert operators._estimate_grid_dimensions(padded, voxel_size_m) == (
        est_width,
        est_height,
        est_depth,
    )
