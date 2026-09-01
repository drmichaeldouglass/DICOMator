"""Static checks on the Blender-facing surface of the add-on.

The headless suite stubs out ``bpy``, so ``bpy.props.*`` returns ``None`` and a
mistyped property name never raises here. Inside Blender it does: a
``layout.prop(props, "typo")`` aborts ``draw()`` and blanks the whole panel,
and a ``bl_parent_id`` pointing at a panel that is not registered leaves that
sub-panel out of the sidebar. Both failures only show up at runtime, so this
module checks the wiring by reading the source instead.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Attributes of a ``PropertyGroup`` that Blender provides, so ``props.<name>``
#: may legitimately reference them without a declaration in the group.
BUILTIN_PROPERTY_GROUP_ATTRIBUTES = frozenset({"bl_rna", "id_data", "name", "rna_type"})

#: Scene-level pointer installed by ``register()``; it lives on ``bpy.types.Scene``
#: rather than on ``bpy.types.Object``, so the per-object checks must skip it.
SCENE_POINTER = "dicomator_props"


def _parse(name: str) -> ast.Module:
    return ast.parse((ROOT / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scene_properties() -> frozenset[str]:
    """Names declared on ``DICOMATOR_PG_properties``."""

    for node in ast.walk(_parse("properties.py")):
        if isinstance(node, ast.ClassDef) and node.name == "DICOMATOR_PG_properties":
            return frozenset(
                entry.target.id
                for entry in node.body
                if isinstance(entry, ast.AnnAssign) and isinstance(entry.target, ast.Name)
            )
    raise AssertionError("DICOMATOR_PG_properties not found in properties.py")


@pytest.fixture(scope="module")
def object_properties() -> frozenset[str]:
    """Names ``register()`` installs on ``bpy.types.Object``."""

    source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    return frozenset(re.findall(r"bpy\.types\.Object\.(dicomator_\w+)\s*=", source))


def _string_argument(call: ast.Call, index: int) -> str | None:
    if len(call.args) <= index:
        return None
    argument = call.args[index]
    return argument.value if isinstance(argument, ast.Constant) and isinstance(argument.value, str) else None


def _first_argument_is_props(call: ast.Call) -> bool:
    return bool(call.args) and isinstance(call.args[0], ast.Name) and call.args[0].id == "props"


def _referenced_scene_properties(module: str) -> list[tuple[int, str]]:
    """Return ``(line, name)`` for every scene property the module touches."""

    found: list[tuple[int, str]] = []
    for node in ast.walk(_parse(module)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "props"
            and node.attr not in BUILTIN_PROPERTY_GROUP_ATTRIBUTES
        ):
            found.append((node.lineno, node.attr))
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        # layout.prop(props, "name") / layout.prop_search(props, "name", ...)
        if (
            isinstance(function, ast.Attribute)
            and function.attr in {"prop", "prop_search"}
            and _first_argument_is_props(node)
            and (name := _string_argument(node, 1))
        ):
            found.append((node.lineno, name))
        # getattr(props, "name", default)
        if (
            isinstance(function, ast.Name)
            and function.id == "getattr"
            and _first_argument_is_props(node)
            and (name := _string_argument(node, 1))
            and name not in BUILTIN_PROPERTY_GROUP_ATTRIBUTES
        ):
            found.append((node.lineno, name))
    return found


@pytest.mark.parametrize("module", ["panels.py", "operators.py", "properties.py"])
def test_referenced_scene_properties_are_declared(module, scene_properties):
    unknown = [
        f"{module}:{line} references props.{name}"
        for line, name in _referenced_scene_properties(module)
        if name not in scene_properties
    ]
    assert not unknown, unknown


@pytest.mark.parametrize("module", ["panels.py", "operators.py", "voxelization.py"])
def test_referenced_object_properties_are_declared(module, object_properties):
    """``dicomator_*`` names must be installed by ``register()``."""

    unknown: list[str] = []
    for node in ast.walk(_parse(module)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        is_layout_prop = isinstance(function, ast.Attribute) and function.attr == "prop"
        is_getattr = isinstance(function, ast.Name) and function.id == "getattr"
        if not (is_layout_prop or is_getattr):
            continue
        name = _string_argument(node, 1)
        if name is None or not name.startswith("dicomator_") or name == SCENE_POINTER:
            continue
        if name not in object_properties:
            unknown.append(f"{module}:{node.lineno} references {name}")
    assert not unknown, unknown


def test_artifact_flag_table_names_real_properties(scene_properties):
    """``ARTIFACT_FLAGS`` drives the UI, the estimate, and the export pipeline."""

    from conftest import load_module

    constants = load_module("constants")
    for flag in constants.ARTIFACT_FLAGS:
        assert flag in scene_properties, flag


def test_suppressible_setting_table_names_real_properties(scene_properties):
    from conftest import load_module

    constants = load_module("constants")
    for _feature, _label, flags in constants.UI_SUPPRESSIBLE_SETTINGS:
        for flag in flags:
            assert flag in scene_properties, flag


def _panel_class_attributes() -> dict[str, dict[str, str]]:
    """Return ``{class_name: {attribute: value}}`` for every panel in panels.py."""

    panels: dict[str, dict[str, str]] = {}
    for node in _parse("panels.py").body:
        if not isinstance(node, ast.ClassDef):
            continue
        attributes: dict[str, str] = {}
        for entry in node.body:
            if (
                isinstance(entry, ast.Assign)
                and len(entry.targets) == 1
                and isinstance(entry.targets[0], ast.Name)
                and isinstance(entry.value, ast.Constant)
                and isinstance(entry.value.value, str)
            ):
                attributes[entry.targets[0].id] = entry.value.value
        if "bl_idname" in attributes:
            panels[node.name] = attributes
    return panels


def test_every_panel_parent_is_a_registered_panel():
    """A dangling ``bl_parent_id`` drops the sub-panel out of the sidebar."""

    panels = _panel_class_attributes()
    registered = {attributes["bl_idname"] for attributes in panels.values()}
    for class_name, attributes in panels.items():
        parent = attributes.get("bl_parent_id")
        if parent is not None:
            assert parent in registered, f"{class_name}.bl_parent_id = {parent!r}"


def test_every_panel_is_registered_and_exported():
    """Panels have to reach both ``classes`` and ``__all__`` to appear in Blender."""

    panels = _panel_class_attributes()
    entry_point = (ROOT / "__init__.py").read_text(encoding="utf-8")
    panels_source = (ROOT / "panels.py").read_text(encoding="utf-8")

    classes_tuple = re.search(r"^classes = \((.*?)\)$", entry_point, re.S | re.M)
    assert classes_tuple is not None
    registered = set(re.findall(r"\w+", classes_tuple.group(1)))

    for class_name in panels:
        assert class_name in registered, f"{class_name} is missing from classes"
        assert f'"{class_name}"' in panels_source, f"{class_name} is missing from __all__"


def test_panels_share_one_sidebar_category():
    """Split categories would scatter the add-on across several sidebar tabs."""

    categories = {
        attributes["bl_category"]
        for attributes in _panel_class_attributes().values()
        if "bl_category" in attributes
    }
    assert categories == {"DICOMator"}


def test_operator_is_registered_and_uses_the_add_on_namespace():
    operators = (ROOT / "operators.py").read_text(encoding="utf-8")
    entry_point = (ROOT / "__init__.py").read_text(encoding="utf-8")

    for idname in re.findall(r'bl_idname = "([\w.]+)"', operators):
        assert idname.startswith("dicomator."), idname

    for class_name in re.findall(r"^class (\w+)\(Operator\)", operators, re.M):
        assert class_name in entry_point, f"{class_name} is never registered"


def test_register_and_unregister_are_symmetric(object_properties):
    """Every property ``register()`` adds must be removed by ``unregister()``."""

    entry_point = (ROOT / "__init__.py").read_text(encoding="utf-8")
    _register, _, unregister = entry_point.partition("def unregister()")
    for name in object_properties | {SCENE_POINTER}:
        assert f"bpy.types.Object.{name}" in unregister or f"bpy.types.Scene.{name}" in unregister, name
