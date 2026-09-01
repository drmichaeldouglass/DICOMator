"""Panel draw tests for the Basic/Intermediate/Advanced interface modes.

The panels cannot run without Blender, so a minimal layout stand-in records
which properties each panel would draw. That is enough to pin what every mode
shows and, more importantly, what it hides.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import load_module

constants = load_module("constants")
panels = load_module("panels")

BASIC = constants.UI_MODE_BASIC
INTERMEDIATE = constants.UI_MODE_INTERMEDIATE
ADVANCED = constants.UI_MODE_ADVANCED


class _Layout:
    """Records what a panel draws instead of building Blender widgets."""

    def __init__(
        self,
        drawn_props: list[str],
        labels: list[str],
        operators: list[str],
        prop_labels: dict[str, list[str]],
    ):
        self.drawn_props = drawn_props
        self.labels = labels
        self.operators = operators
        self.prop_labels = prop_labels
        self.use_property_split = False
        self.use_property_decorate = False

    def _child(self) -> "_Layout":
        return _Layout(self.drawn_props, self.labels, self.operators, self.prop_labels)

    def prop(self, _data, name: str, **kwargs) -> None:
        self.drawn_props.append(name)
        self.prop_labels.setdefault(name, []).append(str(kwargs.get("text", "")))

    def label(self, **kwargs) -> None:
        self.labels.append(str(kwargs.get("text", "")))

    def operator(self, idname: str, **_kwargs) -> None:
        self.operators.append(idname)

    def row(self, **_kwargs) -> "_Layout":
        return self._child()

    def column(self, **_kwargs) -> "_Layout":
        return self._child()

    def box(self, **_kwargs) -> "_Layout":
        return self._child()

    def grid_flow(self, **_kwargs) -> "_Layout":
        return self._child()

    def separator(self, **_kwargs) -> None:
        return None


class _Matrix:
    def __matmul__(self, vector):
        return vector


def _mesh(name: str, object_type: str = "CT", size: float = 0.1) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        type='MESH',
        mode='OBJECT',
        matrix_world=_Matrix(),
        bound_box=[
            (x, y, z)
            for x in (0.0, size)
            for y in (0.0, size)
            for z in (0.0, size)
        ],
        dicomator_object_type=object_type,
        dicomator_material="CUSTOM",
        dicomator_hu=0.0,
        dicomator_priority=0,
        dicomator_dose=0.0,
        dicomator_roi_type="OAR",
    )


def _props(**overrides) -> SimpleNamespace:
    values = {
        "ui_mode": BASIC,
        "export_image_series": True,
        "export_drr": False,
        "export_rtdose": False,
        "export_rtstruct": False,
        "export_4d": False,
        "use_timeline_range": True,
        "frame_start": 1,
        "frame_end": 5,
        "frame_step": 1,
        "lateral_resolution_mm": 2.0,
        "axial_resolution_mm": 2.0,
        "apply_modifiers": True,
        "allow_oversized_grids": False,
        "export_directory": "",
        "imaging_modality": constants.MODALITY_CT,
        "artifact_seed": 0,
        "ring_random_radius": False,
    }
    values.update({flag: False for flag in constants.ARTIFACT_FLAGS})
    values.update(overrides)
    return SimpleNamespace(**values)


def _context(props, objects) -> SimpleNamespace:
    scene = SimpleNamespace(
        dicomator_props=props,
        camera=None,
        frame_start=1,
        frame_end=10,
        frame_current=1,
        unit_settings=SimpleNamespace(scale_length=1.0),
    )
    return SimpleNamespace(
        scene=scene,
        selected_objects=list(objects),
        active_object=objects[0] if objects else None,
    )


def _draw(panel_cls, props, objects, tmp_path, monkeypatch) -> dict[str, object]:
    """Draw ``panel_cls`` and return the props, labels, and operators emitted."""

    monkeypatch.setattr(panels, "ensure_pydicom_available", lambda: True)
    monkeypatch.setattr(panels, "resolve_output_directory", lambda _value: str(tmp_path / "export"))

    drawn_props: list[str] = []
    labels: list[str] = []
    operators: list[str] = []
    prop_labels: dict[str, list[str]] = {}
    panel = panel_cls.__new__(panel_cls)
    panel.layout = _Layout(drawn_props, labels, operators, prop_labels)
    panel.draw(_context(props, objects))
    return {
        "props": drawn_props,
        "labels": labels,
        "operators": operators,
        "prop_labels": prop_labels,
    }


# ---------------------------------------------------------------------------
# Root panel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE, ADVANCED])
def test_every_mode_offers_the_mode_selector_and_export_button(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_panel, _props(ui_mode=mode), [_mesh("Body")], tmp_path, monkeypatch
    )
    assert "ui_mode" in drawn["props"]
    assert drawn["operators"] == ["dicomator.export_dicom"]


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_simple_modes_hide_the_output_checkboxes(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_panel, _props(ui_mode=mode), [_mesh("Body")], tmp_path, monkeypatch
    )
    for name in ("export_image_series", "export_drr", "export_rtdose", "export_rtstruct"):
        assert name not in drawn["props"]


def test_advanced_mode_shows_every_output_checkbox(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_panel, _props(ui_mode=ADVANCED), [_mesh("Body")], tmp_path, monkeypatch
    )
    for name in ("export_image_series", "export_drr", "export_rtdose", "export_rtstruct"):
        assert name in drawn["props"]


def test_basic_mode_reports_settings_it_holds_inactive(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_panel,
        _props(enable_noise=True, export_rtdose=True),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    notice = [text for text in drawn["labels"] if text.startswith("Inactive in Basic mode:")]
    assert notice == ["Inactive in Basic mode: Artifacts, RT Dose"]


def test_simple_modes_flag_non_image_meshes_as_unexported(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_panel,
        _props(),
        [_mesh("Body"), _mesh("Dose", "RTDOSE")],
        tmp_path,
        monkeypatch,
    )
    assert "1 non-image mesh(es) not exported in Basic mode" in drawn["labels"]


# ---------------------------------------------------------------------------
# Objects panel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_simple_modes_hide_the_per_object_dicom_type(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_per_object_hu,
        _props(ui_mode=mode),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert "dicomator_object_type" not in drawn["props"]
    assert "dicomator_material" in drawn["props"]
    assert "dicomator_hu" in drawn["props"]
    assert "imaging_modality" in drawn["props"]


def test_basic_mode_hides_overlap_priority(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_per_object_hu, _props(), [_mesh("Body")], tmp_path, monkeypatch
    )
    assert "dicomator_priority" not in drawn["props"]


@pytest.mark.parametrize("mode", [INTERMEDIATE, ADVANCED])
def test_overlap_priority_returns_with_the_richer_modes(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_per_object_hu,
        _props(ui_mode=mode),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert "dicomator_priority" in drawn["props"]


def test_simple_modes_offer_no_dose_or_roi_settings_for_typed_meshes(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_per_object_hu,
        _props(),
        [_mesh("Dose", "RTDOSE"), _mesh("Heart", "RTSTRUCT")],
        tmp_path,
        monkeypatch,
    )
    assert "dicomator_dose" not in drawn["props"]
    assert "dicomator_roi_type" not in drawn["props"]
    assert drawn["labels"].count("Not exported in Basic mode") == 2


def test_advanced_mode_shows_dose_and_roi_settings(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_per_object_hu,
        _props(ui_mode=ADVANCED),
        [_mesh("Dose", "RTDOSE"), _mesh("Heart", "RTSTRUCT")],
        tmp_path,
        monkeypatch,
    )
    assert "dicomator_object_type" in drawn["props"]
    assert "dicomator_dose" in drawn["props"]
    assert "dicomator_roi_type" in drawn["props"]


# ---------------------------------------------------------------------------
# Export panel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE, ADVANCED])
def test_every_mode_shows_the_core_export_settings(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_export_settings,
        _props(ui_mode=mode),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    for name in ("lateral_resolution_mm", "axial_resolution_mm", "apply_modifiers", "export_directory"):
        assert name in drawn["props"]


def test_basic_mode_hides_4d_and_the_grid_override(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_export_settings, _props(), [_mesh("Body")], tmp_path, monkeypatch
    )
    assert "export_4d" not in drawn["props"]
    assert "allow_oversized_grids" not in drawn["props"]


def test_intermediate_mode_adds_4d_but_not_the_grid_override(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_export_settings,
        _props(ui_mode=INTERMEDIATE, export_4d=True),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert "export_4d" in drawn["props"]
    assert "frame_step" in drawn["props"]
    assert "allow_oversized_grids" not in drawn["props"]


def test_advanced_mode_shows_drr_dose_and_grid_settings(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_export_settings,
        _props(ui_mode=ADVANCED, export_drr=True, export_rtdose=True),
        [_mesh("Body"), _mesh("Dose", "RTDOSE")],
        tmp_path,
        monkeypatch,
    )
    assert "allow_oversized_grids" in drawn["props"]
    assert "drr_resolution_scale" in drawn["props"]
    assert "dose_type" in drawn["props"]
    assert "dose_accumulation" in drawn["props"]


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_simple_modes_hide_dose_settings_even_with_a_dose_mesh(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_export_settings,
        _props(ui_mode=mode, export_rtdose=True),
        [_mesh("Body"), _mesh("Dose", "RTDOSE")],
        tmp_path,
        monkeypatch,
    )
    assert "dose_type" not in drawn["props"]
    assert "drr_resolution_scale" not in drawn["props"]


# ---------------------------------------------------------------------------
# Artifacts panel
# ---------------------------------------------------------------------------


def test_artifacts_panel_is_hidden_in_basic_mode():
    context = _context(_props(), [_mesh("Body")])
    assert not panels.VIEW3D_PT_dicomator_artifacts.poll(context)


@pytest.mark.parametrize("mode", [INTERMEDIATE, ADVANCED])
def test_artifacts_panel_appears_in_the_richer_modes(mode):
    context = _context(_props(ui_mode=mode), [_mesh("Body")])
    assert panels.VIEW3D_PT_dicomator_artifacts.poll(context)


def test_artifacts_panel_stays_hidden_without_an_image_series():
    props = _props(ui_mode=ADVANCED, export_image_series=False, export_drr=True)
    assert not panels.VIEW3D_PT_dicomator_artifacts.poll(_context(props, [_mesh("Body")]))


def test_artifacts_panel_draws_its_ct_toggles(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_artifacts,
        _props(ui_mode=INTERMEDIATE),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert "artifact_seed" in drawn["props"]
    for flag in ("enable_noise", "enable_metal_artifacts", "enable_motion_artifact"):
        assert flag in drawn["props"]


def test_artifacts_panel_names_noise_for_the_selected_modality(tmp_path, monkeypatch):
    ct = _draw(
        panels.VIEW3D_PT_dicomator_artifacts,
        _props(ui_mode=INTERMEDIATE, imaging_modality=constants.MODALITY_CT),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    mr = _draw(
        panels.VIEW3D_PT_dicomator_artifacts,
        _props(ui_mode=INTERMEDIATE, imaging_modality=constants.MODALITY_MRI_T1),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert ct["prop_labels"]["enable_noise"] == ["Gaussian"]
    assert mr["prop_labels"]["enable_noise"] == ["Rician"]


# ---------------------------------------------------------------------------
# Estimate panel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE, ADVANCED])
def test_estimate_panel_draws_in_every_mode(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_selection_info,
        _props(ui_mode=mode),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert any(text.startswith("Est. Grid:") for text in drawn["labels"])


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_estimate_panel_skips_the_drr_camera_hint_outside_advanced(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_selection_info,
        _props(ui_mode=mode, export_drr=True),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert "Set an active scene camera for DRR export" not in drawn["labels"]


def test_estimate_panel_warns_about_a_missing_drr_camera_in_advanced(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_selection_info,
        _props(ui_mode=ADVANCED, export_drr=True),
        [_mesh("Body")],
        tmp_path,
        monkeypatch,
    )
    assert "Set an active scene camera for DRR export" in drawn["labels"]


@pytest.mark.parametrize("mode", [BASIC, INTERMEDIATE])
def test_estimate_ignores_non_image_meshes_in_simple_modes(mode, tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_selection_info,
        _props(ui_mode=mode),
        [_mesh("Body"), _mesh("Far Dose", "RTDOSE", size=100.0)],
        tmp_path,
        monkeypatch,
    )
    assert "Est. Grid: 52 x 52 x 52" in drawn["labels"]


def test_export_action_ignores_meshes_for_disabled_outputs(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_panel,
        _props(
            ui_mode=ADVANCED,
            export_image_series=True,
            export_rtdose=False,
        ),
        [_mesh("Body"), _mesh("Far Dose", "RTDOSE", size=100.0)],
        tmp_path,
        monkeypatch,
    )
    assert "Grid too large - export will abort" not in drawn["labels"]
    assert drawn["operators"] == ["dicomator.export_dicom"]


def test_export_action_hides_button_for_a_blocked_grid(tmp_path, monkeypatch):
    drawn = _draw(
        panels.VIEW3D_PT_dicomator_panel,
        _props(),
        [_mesh("Huge Body", size=100.0)],
        tmp_path,
        monkeypatch,
    )
    assert "Grid too large - export will abort" in drawn["labels"]
    assert drawn["operators"] == []
