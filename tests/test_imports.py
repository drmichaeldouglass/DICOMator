"""Smoke test: every add-on submodule must import headlessly.

This catches syntax errors, bad imports, and signature typos in the
Blender-facing modules (operators/panels/properties) that the functional
tests do not otherwise execute.
"""
from __future__ import annotations

import pytest

from conftest import load_module

MODULES = [
    "constants",
    "utils",
    "artifacts",
    "dicom_export",
    "drr",
    "rtdose_export",
    "rtstruct_export",
    "voxelization",
    "properties",
    "operators",
    "panels",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    module = load_module(name)
    assert module is not None
