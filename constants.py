"""Constants and optional dependencies for the DICOMator add-on."""
from __future__ import annotations

import importlib

AIR_DENSITY = -1000.0  # HU value for air (DICOM standard reference for air)
DEFAULT_DENSITY = 0.0   # Default HU for objects unless overridden per-object
MAX_HU_VALUE = 3071     # Max HU for 12-bit CT representations
MIN_HU_VALUE = -1024    # Min HU (typical CT lower bound)

PYDICOM_AVAILABLE = False
Dataset = None
FileDataset = None
generate_uid = None
pydicom = None

_pydicom_spec = importlib.util.find_spec("pydicom")
if _pydicom_spec is not None:
    pydicom = importlib.import_module("pydicom")
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid
    PYDICOM_AVAILABLE = True
else:
    print("Warning: pydicom not available. DICOM export functionality will be disabled.")

__all__ = [
    "AIR_DENSITY",
    "DEFAULT_DENSITY",
    "MAX_HU_VALUE",
    "MIN_HU_VALUE",
    "PYDICOM_AVAILABLE",
    "Dataset",
    "FileDataset",
    "generate_uid",
    "pydicom",
]
