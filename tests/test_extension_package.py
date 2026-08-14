"""Checks for official Blender Extensions package requirements."""
from __future__ import annotations

import hashlib
import re
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "blender_manifest.toml"


def _manifest() -> dict:
    with MANIFEST_PATH.open("rb") as handle:
        return tomllib.load(handle)


def test_manifest_has_submission_metadata():
    manifest = _manifest()

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["id"] == "dicomator"
    assert manifest["type"] == "add-on"
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"])
    assert len(manifest["tagline"]) <= 64
    assert manifest["license"] == ["SPDX:GPL-3.0-or-later"]
    assert manifest["website"] == "https://extensions.blender.org/add-ons/dicomator/"
    assert manifest["permissions"] == {
        "files": "Write DICOM exports to user-selected folders"
    }


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
    wheel = ROOT / "wheels" / "pydicom-3.0.1-py3-none-any.whl"

    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == (
        "db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41"
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
