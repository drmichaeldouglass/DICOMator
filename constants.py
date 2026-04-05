"""Constants and optional dependencies for the DICOMator add-on."""
from __future__ import annotations

import importlib
import logging
import sys
import zipfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

AIR_DENSITY = -1000.0  # HU value for air (DICOM standard reference for air)
DEFAULT_DENSITY = 0.0   # Default HU for objects unless overridden per-object
MAX_HU_VALUE = 3071     # Max HU for 12-bit CT representations
MIN_HU_VALUE = -1024    # Min HU (typical CT lower bound)

# Imaging modality identifiers used when mapping tissue presets to intensities.
MODALITY_CT = "CT"
MODALITY_MRI_T1 = "MRI_T1"
MODALITY_MRI_T2 = "MRI_T2"
OUTPUT_MODE_VOLUME = "VOLUME"
OUTPUT_MODE_DRR = "DRR"

MRI_MODALITIES = {MODALITY_MRI_T1, MODALITY_MRI_T2}

IMAGING_MODALITY_ITEMS = [
    (MODALITY_CT, "CT", "Assign CT Hounsfield Units"),
    (MODALITY_MRI_T1, "T1 MR", "Assign intensities for T1-weighted MRI"),
    (MODALITY_MRI_T2, "T2 MR", "Assign intensities for T2-weighted MRI"),
]

OUTPUT_MODE_ITEMS = [
    (OUTPUT_MODE_VOLUME, "Synthetic Volume", "Export a voxelized CT or MR image series"),
    (OUTPUT_MODE_DRR, "DRR", "Export a digital reconstructed radiograph from the active camera"),
]

# Tissue/material presets with representative intensities for each modality.
MATERIAL_INTENSITIES = {
    "AIR": {
        MODALITY_CT: -1000,
        MODALITY_MRI_T1: 0,
        MODALITY_MRI_T2: 0,
    },
    "CORTICAL_BONE": {
        MODALITY_CT: 1100,
        MODALITY_MRI_T1: 10,
        MODALITY_MRI_T2: 8,
    },
    "TRABECULAR_BONE": {
        MODALITY_CT: 200,
        MODALITY_MRI_T1: 190,
        MODALITY_MRI_T2: 170,
    },
    "FAT": {
        MODALITY_CT: -75,
        MODALITY_MRI_T1: 210,
        MODALITY_MRI_T2: 170,
    },
    "MUSCLE": {
        MODALITY_CT: 50,
        MODALITY_MRI_T1: 90,
        MODALITY_MRI_T2: 80,
    },
    "LIVER": {
        MODALITY_CT: 50,
        MODALITY_MRI_T1: 100,
        MODALITY_MRI_T2: 90,
    },
    "SPLEEN": {
        MODALITY_CT: 50,
        MODALITY_MRI_T1: 110,
        MODALITY_MRI_T2: 120,
    },
    "KIDNEY_CORTEX": {
        MODALITY_CT: 40,
        MODALITY_MRI_T1: 110,
        MODALITY_MRI_T2: 100,
    },
    "KIDNEY_MEDULLA": {
        MODALITY_CT: 40,
        MODALITY_MRI_T1: 90,
        MODALITY_MRI_T2: 150,
    },
    "CARTILAGE": {
        MODALITY_CT: 200,
        MODALITY_MRI_T1: 100,
        MODALITY_MRI_T2: 130,
    },
    "BLOOD_ACUTE": {
        MODALITY_CT: 60,
        MODALITY_MRI_T1: 130,
        MODALITY_MRI_T2: 30,
    },
    "WHITE_MATTER": {
        MODALITY_CT: 25,
        MODALITY_MRI_T1: 150,
        MODALITY_MRI_T2: 50,
    },
    "GRAY_MATTER": {
        MODALITY_CT: 40,
        MODALITY_MRI_T1: 110,
        MODALITY_MRI_T2: 110,
    },
    "CSF_WATER": {
        MODALITY_CT: 0,
        MODALITY_MRI_T1: 30,
        MODALITY_MRI_T2: 230,
    },
    "LUNG": {
        MODALITY_CT: -700,       # typical aerated lung parenchyma
        MODALITY_MRI_T1: 20,
        MODALITY_MRI_T2: 80,
    },
    "SOFT_TISSUE": {
        MODALITY_CT: 40,         # generic soft tissue (~muscle)
        MODALITY_MRI_T1: 100,
        MODALITY_MRI_T2: 90,
    },
    "ALUMINIUM": {
        MODALITY_CT: 300,        # moderately dense metal equivalent
        MODALITY_MRI_T1: 0,      # metal causes signal void in MRI
        MODALITY_MRI_T2: 0,
    },
    "TITANIUM": {
        MODALITY_CT: 3000,       # very high CT attenuation (may clip to MAX_HU_VALUE)
        MODALITY_MRI_T1: 0,      # signal void / artifact in MRI
        MODALITY_MRI_T2: 0,
    },
}

MATERIAL_ITEMS = [
    ("CUSTOM", "Custom", "Manually specify an intensity value"),
    ("AIR", "Air", "No signal"),
    ("CORTICAL_BONE", "Cortical Bone / Calcification", "Very dense bone"),
    ("TRABECULAR_BONE", "Trabecular Bone / Fatty Marrow", "Fat-rich cancellous bone"),
    ("FAT", "Fat (subcutaneous, orbital)", "Fat signal"),
    ("MUSCLE", "Muscle", "Intermediate intensity muscle"),
    ("LIVER", "Liver", "Intermediate liver signal"),
    ("SPLEEN", "Spleen", "Slightly brighter than liver on T2"),
    ("KIDNEY_CORTEX", "Kidney Cortex", "Outer renal cortex"),
    ("KIDNEY_MEDULLA", "Kidney Medulla", "Inner renal medulla"),
    ("CARTILAGE", "Cartilage", "Intermediate-bright cartilage"),
    ("BLOOD_ACUTE", "Blood (acute, deoxyHb)", "Acute blood signal"),
    ("WHITE_MATTER", "White Matter", "Brighter than gray on T1"),
    ("GRAY_MATTER", "Gray Matter", "Brighter than white on T2"),
    ("CSF_WATER", "CSF / Water / Edema", "Fluid signal"),
    ("LUNG", "Lung Parenchyma", "Aerated lung tissue"),
    ("SOFT_TISSUE", "Soft Tissue", "Generic soft tissue / organ parenchyma"),
    ("ALUMINIUM", "Aluminium", "Moderately dense metal (implant/foil)"),
    ("TITANIUM", "Titanium (implant)", "High-density metal, produces CT hyperintensity"),
]


def get_material_intensity(material_key: str, modality: str) -> float | None:
    """Return the representative intensity for ``material_key`` in ``modality``."""

    intensities = MATERIAL_INTENSITIES.get(material_key)
    if not intensities:
        return None
    return intensities.get(modality)


# ---------------------------------------------------------------------------
# RT DICOM SOP class UIDs
# ---------------------------------------------------------------------------

#: SOP Class UID for RT Structure Set (DICOM PS3.4 B.5)
RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"

#: SOP Class UID for RT Dose (DICOM PS3.4 B.5)
RTDOSE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"

# ---------------------------------------------------------------------------
# Per-object DICOM type items (used as EnumProperty items on bpy.types.Object)
# ---------------------------------------------------------------------------

DICOM_OBJECT_TYPE_ITEMS = [
    ("CT", "CT Volume", "Voxelize and export as a CT image series"),
    ("RTDOSE", "RT Dose", "Voxelize and export as an RT Dose grid (Gy)"),
    ("RTSTRUCT", "RT Structure", "Export surface contours as an RT Structure Set"),
]

# ---------------------------------------------------------------------------
# ROI type items (DICOM RTROIInterpretedType codes, PS3.3 C.8.8.8)
# ---------------------------------------------------------------------------

ROI_TYPE_ITEMS = [
    ("GTV", "GTV", "Gross Tumour Volume"),
    ("CTV", "CTV", "Clinical Target Volume"),
    ("PTV", "PTV", "Planning Target Volume"),
    ("OAR", "OAR", "Organ At Risk"),
    ("EXTERNAL", "External", "External patient outline / body contour"),
    ("CONTROL", "Control", "Control ROI (e.g. dose normalisation point)"),
    ("AVOIDANCE", "Avoidance", "Region to avoid during optimisation"),
    ("ORGAN", "Organ", "Anatomical organ not classified as an OAR"),
    ("TREATED_VOLUME", "Treated Volume", "Treated volume (isodose surface)"),
    ("IRRAD_VOLUME", "Irradiated Volume", "Volume receiving a clinically significant dose"),
]

# ---------------------------------------------------------------------------
# RT Dose metadata items
# ---------------------------------------------------------------------------

DOSE_TYPE_ITEMS = [
    ("PHYSICAL", "Physical", "Physical absorbed dose (Gy)"),
    ("EFFECTIVE", "Effective", "Radiobiologically weighted effective dose"),
]

DOSE_SUMMATION_TYPE_ITEMS = [
    ("PLAN", "Plan", "Summed over all beams in the plan"),
    ("FRACTION", "Fraction", "Dose for a single treatment fraction"),
    ("BEAM", "Beam", "Dose from a single beam"),
]

PYDICOM_AVAILABLE = False
Dataset = None
FileDataset = None
generate_uid = None
pydicom = None
PYDICOM_IMPORT_ERROR = ""


def ensure_pydicom_available() -> bool:
    """Import ``pydicom`` on demand and cache the resolved module globals."""

    global PYDICOM_AVAILABLE
    global PYDICOM_IMPORT_ERROR
    global Dataset
    global FileDataset
    global generate_uid
    global pydicom

    if PYDICOM_AVAILABLE and pydicom is not None and Dataset is not None and FileDataset is not None and generate_uid is not None:
        return True

    # Wheels are zip archives; pydicom uses open() on __file__-relative paths
    # (e.g. data/urls.json), which fails when the module lives inside a zip.
    # Extract the wheel to a real directory alongside the wheel file so that
    # normal filesystem I/O works correctly.
    _wheels_dir = Path(__file__).parent / "wheels"
    if _wheels_dir.is_dir():
        for _whl in _wheels_dir.glob("pydicom*.whl"):
            _extract_dir = _whl.parent / (_whl.stem + "_extracted")
            if not _extract_dir.is_dir():
                with zipfile.ZipFile(_whl) as _zf:
                    _zf.extractall(_extract_dir)
            _extract_str = str(_extract_dir)
            if _extract_str not in sys.path:
                sys.path.insert(0, _extract_str)

    try:
        module = importlib.import_module("pydicom")
        dataset_module = importlib.import_module("pydicom.dataset")
        uid_module = importlib.import_module("pydicom.uid")

        pydicom = module
        Dataset = dataset_module.Dataset
        FileDataset = dataset_module.FileDataset
        generate_uid = uid_module.generate_uid
        PYDICOM_AVAILABLE = True
        PYDICOM_IMPORT_ERROR = ""
        return True
    except Exception as exc:
        PYDICOM_AVAILABLE = False
        PYDICOM_IMPORT_ERROR = str(exc)
        pydicom = None
        Dataset = None
        FileDataset = None
        generate_uid = None
        LOGGER.warning(
            "pydicom not available or failed to import. DICOM export functionality will be disabled.",
            exc_info=True,
        )
        return False


def get_pydicom_error() -> str:
    """Return the last pydicom import error, if any."""

    return str(PYDICOM_IMPORT_ERROR or "")


ensure_pydicom_available()

__all__ = [
    "AIR_DENSITY",
    "DEFAULT_DENSITY",
    "MAX_HU_VALUE",
    "MIN_HU_VALUE",
    "MODALITY_CT",
    "MODALITY_MRI_T1",
    "MODALITY_MRI_T2",
    "OUTPUT_MODE_VOLUME",
    "OUTPUT_MODE_DRR",
    "MRI_MODALITIES",
    "IMAGING_MODALITY_ITEMS",
    "OUTPUT_MODE_ITEMS",
    "MATERIAL_INTENSITIES",
    "MATERIAL_ITEMS",
    "get_material_intensity",
    "RTSTRUCT_SOP_CLASS",
    "RTDOSE_SOP_CLASS",
    "DICOM_OBJECT_TYPE_ITEMS",
    "ROI_TYPE_ITEMS",
    "DOSE_TYPE_ITEMS",
    "DOSE_SUMMATION_TYPE_ITEMS",
    "ensure_pydicom_available",
    "get_pydicom_error",
    "PYDICOM_AVAILABLE",
    "Dataset",
    "FileDataset",
    "generate_uid",
    "pydicom",
]
