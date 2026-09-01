"""Tests for the Basic/Intermediate/Advanced interface modes.

Each mode hides part of the UI. A hidden setting keeps its stored value, so
these tests pin the other half of that contract: the export pipeline must not
act on a setting the current mode does not show.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import load_module

constants = load_module("constants")
operators = load_module("operators")
panels = load_module("panels")

BASIC = constants.UI_MODE_BASIC
INTERMEDIATE = constants.UI_MODE_INTERMEDIATE
ADVANCED = constants.UI_MODE_ADVANCED


def _props(**overrides) -> SimpleNamespace:
    """Return a property stand-in with every mode-gated flag defined."""

    values = {
        "ui_mode": BASIC,
        "export_image_series": True,
        "export_drr": False,
        "export_rtdose": False,
        "export_rtstruct": False,
        "export_4d": False,
        "use_timeline_range": False,
        "frame_start": 1,
        "frame_end": 5,
        "frame_step": 1,
        "allow_oversized_grids": False,
        "imaging_modality": constants.MODALITY_CT,
        "artifact_seed": 0,
        "noise_std_dev_hu": 20.0,
        "poisson_scale": 150.0,
    }
    values.update({flag: False for flag in constants.ARTIFACT_FLAGS})
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# Mode table
# ---------------------------------------------------------------------------


def test_every_mode_item_has_a_feature_set_and_label():
    identifiers = [identifier for identifier, _label, _description in constants.UI_MODE_ITEMS]
    assert identifiers == [BASIC, INTERMEDIATE, ADVANCED]
    assert set(identifiers) == set(constants.UI_MODE_FEATURES)
    assert set(identifiers) == set(constants.UI_MODE_LABELS)


def test_modes_are_nested_supersets():
    """Each mode must reveal settings, never take previously shown ones away."""

    basic = constants.UI_MODE_FEATURES[BASIC]
    intermediate = constants.UI_MODE_FEATURES[INTERMEDIATE]
    advanced = constants.UI_MODE_FEATURES[ADVANCED]
    assert basic < intermediate < advanced


def test_basic_hides_every_gated_feature():
    assert constants.UI_MODE_FEATURES[BASIC] == frozenset()


def test_intermediate_adds_artifacts_and_4d_only():
    added = constants.UI_MODE_FEATURES[INTERMEDIATE] - constants.UI_MODE_FEATURES[BASIC]
    assert constants.UI_FEATURE_ARTIFACTS in added
    assert constants.UI_FEATURE_FOUR_D in added
    assert constants.UI_FEATURE_DRR not in added
    assert constants.UI_FEATURE_RT_DOSE not in added
    assert constants.UI_FEATURE_RT_STRUCT not in added


@pytest.mark.parametrize(
    "feature",
    [
        constants.UI_FEATURE_ARTIFACTS,
        constants.UI_FEATURE_FOUR_D,
        constants.UI_FEATURE_DRR,
        constants.UI_FEATURE_RT_DOSE,
        constants.UI_FEATURE_RT_STRUCT,
        constants.UI_FEATURE_OBJECT_PRIORITY,
        constants.UI_FEATURE_OVERSIZED_GRIDS,
    ],
)
def test_advanced_shows_every_feature(feature):
    assert constants.ui_feature_visible(_props(ui_mode=ADVANCED), feature)


def test_unknown_or_missing_mode_falls_back_to_advanced():
    """Property stand-ins predating the selector keep the full feature set."""

    assert constants.normalize_ui_mode(None) == ADVANCED
    assert constants.normalize_ui_mode("NOT_A_MODE") == ADVANCED
    assert constants.ui_mode_of(SimpleNamespace()) == ADVANCED
    assert constants.normalize_ui_mode("basic") == BASIC


# ---------------------------------------------------------------------------
# Output resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_simple_modes_export_only_an_image_series(mode):
    props = _props(
        ui_mode=mode,
        export_drr=True,
        export_rtdose=True,
        export_rtstruct=True,
    )
    assert constants.resolve_export_outputs(props) == {
        "image_series": True,
        "drr": False,
        "rtdose": False,
        "rtstruct": False,
    }


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_simple_modes_still_export_with_the_image_toggle_off(mode):
    """The Image checkbox is hidden, so a stored 'off' must not empty the export."""

    props = _props(ui_mode=mode, export_image_series=False)
    assert constants.resolve_export_outputs(props)["image_series"]


def test_advanced_honours_every_output_toggle():
    props = _props(
        ui_mode=ADVANCED,
        export_image_series=False,
        export_drr=True,
        export_rtdose=True,
        export_rtstruct=True,
    )
    assert constants.resolve_export_outputs(props) == {
        "image_series": False,
        "drr": True,
        "rtdose": True,
        "rtstruct": True,
    }


def test_only_advanced_offers_the_output_selector():
    assert not constants.export_outputs_selectable(_props(ui_mode=BASIC))
    assert not constants.export_outputs_selectable(_props(ui_mode=INTERMEDIATE))
    assert constants.export_outputs_selectable(_props(ui_mode=ADVANCED))


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def test_basic_mode_runs_no_artifact_stages():
    props = _props(ui_mode=BASIC, enable_noise=True, enable_ring_artifacts=True)
    assert operators._configured_artifact_stages(props) == []
    assert not constants.artifacts_enabled_for_props(props)


@pytest.mark.parametrize("mode", [INTERMEDIATE, ADVANCED])
def test_artifact_modes_build_the_configured_stages(mode):
    props = _props(ui_mode=mode, enable_noise=True, enable_ring_artifacts=True)
    assert len(operators._configured_artifact_stages(props)) == 2
    assert constants.artifacts_enabled_for_props(props)


def test_basic_mode_memory_estimate_drops_artifact_buffers():
    """The estimate must describe the pipeline the mode will really run."""

    basic = constants.estimate_peak_memory_bytes_for_props(1000, _props(enable_noise=True))
    lean = constants.estimate_peak_memory_bytes_for_props(1000, _props())
    advanced = constants.estimate_peak_memory_bytes_for_props(
        1000, _props(ui_mode=ADVANCED, enable_noise=True)
    )
    assert basic == lean < advanced


# ---------------------------------------------------------------------------
# 4D export
# ---------------------------------------------------------------------------


class _Scene:
    def __init__(self, current=7, start=1, end=3):
        self.frame_current = current
        self.frame_start = start
        self.frame_end = end


def test_basic_mode_exports_the_current_frame_only():
    context = SimpleNamespace(scene=_Scene())
    props = _props(ui_mode=BASIC, export_4d=True, frame_start=1, frame_end=5)
    assert operators._frame_sequence(context, props) == [7]
    assert not constants.four_d_export_enabled(props)


@pytest.mark.parametrize("mode", [INTERMEDIATE, ADVANCED])
def test_4d_modes_export_the_configured_range(mode):
    context = SimpleNamespace(scene=_Scene())
    props = _props(ui_mode=mode, export_4d=True, frame_start=1, frame_end=5, frame_step=2)
    assert operators._frame_sequence(context, props) == [1, 3, 5]
    assert constants.four_d_export_enabled(props)


# ---------------------------------------------------------------------------
# Grid guardrails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_simple_modes_keep_the_grid_guardrails(mode):
    """'Allow Oversized Grids' is an Advanced control; hiding it re-arms it."""

    assert not constants.oversized_grids_allowed(_props(ui_mode=mode, allow_oversized_grids=True))


def test_advanced_mode_can_waive_the_grid_guardrails():
    assert constants.oversized_grids_allowed(_props(ui_mode=ADVANCED, allow_oversized_grids=True))


# ---------------------------------------------------------------------------
# Suppressed-setting reporting
# ---------------------------------------------------------------------------


def test_default_settings_report_nothing_as_suppressed():
    assert constants.suppressed_feature_labels(_props()) == []


def test_basic_mode_names_every_setting_it_holds_inactive():
    props = _props(
        enable_noise=True,
        export_4d=True,
        export_drr=True,
        export_rtdose=True,
        export_rtstruct=True,
        allow_oversized_grids=True,
    )
    assert constants.suppressed_feature_labels(props) == [
        "Artifacts",
        "4D export",
        "DRR",
        "RT Dose",
        "RT Structure",
        "Oversized grids",
    ]


def test_intermediate_mode_keeps_artifacts_and_4d_active():
    props = _props(
        ui_mode=INTERMEDIATE,
        enable_noise=True,
        export_4d=True,
        export_rtdose=True,
    )
    assert constants.suppressed_feature_labels(props) == ["RT Dose"]


def test_advanced_mode_suppresses_nothing():
    props = _props(
        ui_mode=ADVANCED,
        enable_noise=True,
        export_4d=True,
        export_drr=True,
        export_rtdose=True,
        export_rtstruct=True,
        allow_oversized_grids=True,
    )
    assert constants.suppressed_feature_labels(props) == []


def test_suppressible_settings_cover_every_gated_feature():
    """A newly gated feature must be reportable, not silently dropped."""

    reported = {feature for feature, _label, _flags in constants.UI_SUPPRESSIBLE_SETTINGS}
    gated = set(constants.UI_MODE_FEATURES[ADVANCED])
    # Overlap priority is a per-object property, so it has no scene-level flag
    # to report as inactive.
    assert gated - reported == {constants.UI_FEATURE_OBJECT_PRIORITY}


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def test_panel_mode_label_matches_the_enum_item():
    assert panels._mode_label(_props(ui_mode=BASIC)) == "Basic"
    assert panels._mode_label(_props(ui_mode=INTERMEDIATE)) == "Intermediate"
    assert panels._mode_label(_props(ui_mode=ADVANCED)) == "Advanced"
