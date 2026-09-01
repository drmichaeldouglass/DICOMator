"""Checks for official Blender Extensions package requirements."""
from __future__ import annotations

import fnmatch
import hashlib
import re
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "blender_manifest.toml"

#: Repository entries that must end up inside the built extension package.
#: Anything else in the repository root has to be covered by a
#: ``paths_exclude_pattern`` entry, or it is shipped to every user.
PACKAGED_ENTRIES = frozenset({
    "LICENSE",
    "README.md",
    "__init__.py",
    "artifacts.py",
    "blender_manifest.toml",
    "constants.py",
    "dicom_export.py",
    "drr.py",
    "operators.py",
    "panels.py",
    "properties.py",
    "rtdose_export.py",
    "rtstruct_export.py",
    "utils.py",
    "voxelization.py",
    "wheels",
})


def _manifest() -> dict:
    with MANIFEST_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _exclude_patterns() -> list[str]:
    return _manifest().get("build", {}).get("paths_exclude_pattern", [])


def _excludes_root_entry(pattern: str, name: str, is_dir: bool) -> bool:
    """Return True when ``pattern`` hides a repository-root entry.

    Blender matches ``paths_exclude_pattern`` with gitignore semantics. At the
    repository root that reduces to: a leading ``/`` only anchors the pattern
    (top-level entries are already anchored), and a trailing ``/`` restricts
    the pattern to directories. :func:`test_exclude_patterns_use_supported_forms`
    keeps the manifest to the pattern shapes this simplification covers.
    """

    body = pattern[1:] if pattern.startswith("/") else pattern
    if body.endswith("/"):
        if not is_dir:
            return False
        body = body[:-1]
    return fnmatch.fnmatch(name, body)


def test_manifest_has_submission_metadata():
    manifest = _manifest()

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["id"] == "dicomator"
    assert manifest["type"] == "add-on"
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"])
    assert len(manifest["tagline"]) <= 64
    assert manifest["license"] == ["SPDX:GPL-3.0-or-later"]
    assert manifest["website"].startswith("https://")
    assert manifest["permissions"] == {
        "files": "Write DICOM exports to user-selected folders"
    }


def test_pure_python_wheels_are_not_restricted_to_a_platform_list():
    """A ``py3-none-any`` wheel runs everywhere, so no platform may be excluded.

    Declaring ``platforms`` hides the extension from every Blender platform
    left off the list (linux-arm64 and windows-arm64 among them) for no reason
    when nothing in the package is platform specific.
    """

    manifest = _manifest()

    pure_python = all(
        Path(wheel).name.endswith("-py3-none-any.whl") for wheel in manifest["wheels"]
    )
    assert pure_python, "a platform-specific wheel now needs a 'platforms' list"
    assert "platforms" not in manifest


def test_exclude_patterns_use_supported_forms():
    """Keep the manifest to the pattern shapes the root-entry test understands."""

    for pattern in _exclude_patterns():
        assert not pattern.startswith("!"), pattern
        assert "**" not in pattern, pattern
        assert "/" not in pattern.strip("/"), pattern


def test_only_the_add_on_itself_is_packaged():
    """Every repository-root entry either ships or is explicitly excluded.

    Supplying ``paths_exclude_pattern`` replaces Blender's built-in default
    exclude list, so a new development file (or a tool cache such as
    ``.pytest_cache/``) is shipped to users until it is named in the manifest.
    """

    patterns = _exclude_patterns()
    unclassified = []
    for entry in ROOT.iterdir():
        if entry.name in PACKAGED_ENTRIES:
            continue
        if any(_excludes_root_entry(p, entry.name, entry.is_dir()) for p in patterns):
            continue
        unclassified.append(entry.name)

    assert not unclassified, (
        "these paths would ship inside the extension package; add them to "
        f"PACKAGED_ENTRIES or to paths_exclude_pattern: {sorted(unclassified)}"
    )


@pytest.mark.parametrize(
    "cache_dir",
    [".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", ".git", ".github"],
)
def test_developer_caches_are_excluded_from_the_package(cache_dir):
    """Running the test suite before a build must not pollute the package."""

    patterns = _exclude_patterns()
    assert any(_excludes_root_entry(p, cache_dir, True) for p in patterns), cache_dir


def test_every_packaged_entry_exists():
    for name in PACKAGED_ENTRIES:
        assert (ROOT / name).exists(), name


def test_manifest_wheels_exist_and_are_valid_archives():
    manifest = _manifest()

    assert manifest["wheels"]
    for wheel_path in manifest["wheels"]:
        wheel = ROOT / wheel_path
        assert wheel.is_file(), wheel
        with zipfile.ZipFile(wheel) as archive:
            assert archive.testzip() is None
            assert any(name.endswith(".dist-info/METADATA") for name in archive.namelist())


def test_bundled_pydicom_wheel_matches_pypi():
    wheel = ROOT / "wheels" / "pydicom-3.0.2-py3-none-any.whl"

    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == (
        "abf971a5440f84dbaf42c4b6758e30e62480902584f8b270b9a5d146e278a07b"
    )


def test_extension_uses_current_metadata_and_namespaces():
    entry_point = (ROOT / "__init__.py").read_text(encoding="utf-8")
    operators = (ROOT / "operators.py").read_text(encoding="utf-8")
    constants = (ROOT / "constants.py").read_text(encoding="utf-8")

    assert "bl_info" not in entry_point
    assert 'bl_idname = "dicomator.export_dicom"' in operators
    assert "sys.path" not in constants
    assert "extractall" not in constants


def test_repository_contains_gpl_v3_or_later_license():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "SPDX-License-Identifier: GPL-3.0-or-later" in license_text
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 29 June 2007" in license_text
