"""Constants and optional dependencies for the DICOMator add-on."""
from __future__ import annotations

import importlib
import logging
import math
from datetime import datetime

import numpy as np

LOGGER = logging.getLogger(__name__)

AIR_DENSITY = -1000.0  # HU value for air (DICOM standard reference for air)
DEFAULT_DENSITY = 0.0   # Default HU for objects unless overridden per-object
MAX_HU_VALUE = 3071     # Max HU for 12-bit CT representations
MIN_HU_VALUE = -1024    # Min HU (typical CT lower bound)

# Imaging modality identifiers used when mapping tissue presets to intensities.
MODALITY_CT = "CT"
MODALITY_MRI_T1 = "MRI_T1"
MODALITY_MRI_T2 = "MRI_T2"

MRI_MODALITIES = {MODALITY_MRI_T1, MODALITY_MRI_T2}

IMAGING_MODALITY_ITEMS = [
    (MODALITY_CT, "CT", "Assign CT Hounsfield Units"),
    (MODALITY_MRI_T1, "T1 MR", "Assign intensities for T1-weighted MRI"),
    (MODALITY_MRI_T2, "T2 MR", "Assign intensities for T2-weighted MRI"),
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
# MR acquisition parameters per weighting preset
# ---------------------------------------------------------------------------

#: Representative spin-echo acquisition parameters emitted in the MR Image
#: module for each weighting preset: a conventional SE for T1 and a fast
#: (segmented k-space) SE for T2, so the metadata matches the intensity
#: preset the user selected.
MR_SEQUENCE_PARAMETERS = {
    MODALITY_MRI_T1: {
        "RepetitionTime": "500",
        "EchoTime": "15",
        "EchoTrainLength": "1",
        "SequenceVariant": "NONE",
    },
    MODALITY_MRI_T2: {
        "RepetitionTime": "4000",
        "EchoTime": "100",
        "EchoTrainLength": "16",
        "SequenceVariant": "SK",
    },
}


def normalize_dicom_date(value: str | None) -> str:
    """Return ``value`` as an 8-digit DICOM DA string, or '' when invalid.

    Non-digit separators are stripped (e.g. ``1980-02-01`` → ``19800201``);
    anything that does not reduce to exactly 8 digits yields an empty string,
    which is valid for Type 2 date attributes such as PatientBirthDate.
    """

    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) != 8:
        return ""
    try:
        datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return ""
    return digits


def sanitize_dicom_text(value: str | None, default: str = "") -> str:
    """Return a single safe DICOM text value.

    DICOM uses a backslash to separate values. Control characters and a user
    supplied backslash are therefore replaced with spaces before writing a
    single-valued text attribute.
    """

    text = str(value) if value is not None else ""
    text = "".join(
        " " if ch == "\\" or ord(ch) < 32 or 0x7F <= ord(ch) <= 0x9F else ch
        for ch in text
    )
    text = " ".join(text.split())
    return text or str(default)


def truncate_dicom_text(value: str | None, max_bytes: int, default: str = "") -> str:
    """Sanitize and truncate a UTF-8 DICOM value to ``max_bytes``."""

    text = sanitize_dicom_text(value, default)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def format_ds(value: float) -> str:
    """Return ``value`` as a DICOM DS (Decimal String) of at most 16 bytes.

    DICOM limits every Decimal String value to 16 characters (PS3.5 Table
    6.2-1), but Python's shortest round-trip float representation regularly
    needs more: a voxel centre at -122.75 mm reprs as ``-122.74999999999999``
    (19 characters) and a 1.3 mm pixel spacing as ``1.3000000000000003`` (18).
    Writing those verbatim produces files that strict parsers and treatment
    planning systems reject, so the value is re-formatted here to the most
    precise fixed-point (or, for very small/large magnitudes, scientific)
    representation that fits. Roughly 14 significant digits survive, which is
    far more than millimetre geometry needs.
    """

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("DS values must be finite")

    text = repr(number)
    if len(text) <= 16:
        return text

    sign_chars = 1 if number < 0.0 else 0
    exponent = math.log10(abs(number))
    # Below 1e-4 the fixed-point form spends its budget on leading zeros, and
    # above ~1e14 it can no longer hold a fractional digit; both are better
    # served by scientific notation.
    if exponent < -4.0 or exponent >= (14 - sign_chars):
        digits = 10 - sign_chars
        scientific = f"{number:.{digits}e}"
        if len(scientific) > 16:
            # Three-digit exponents (e.g. 1e-200) need one more character.
            scientific = f"{number:.{digits - 1}e}"
        return scientific

    integer_digits = int(math.floor(exponent)) if exponent >= 1.0 else 0
    decimals = max(0, 14 - sign_chars - integer_digits)
    fixed = f"{number:.{decimals}f}"
    while len(fixed) > 16 and decimals > 0:
        # Rounding can carry into an extra integer digit, e.g. 9.999999999999998
        # formats as '10.00000000000000'; give the carry its character back.
        decimals = max(0, decimals - (len(fixed) - 16))
        fixed = f"{number:.{decimals}f}"
    return fixed


def format_ds_sequence(values) -> list[str]:
    """Return ``values`` as a list of conformant DS strings."""

    return [format_ds(value) for value in values]


def truncate_sh(value: str | None, default: str = "") -> str:
    """Clamp ``value`` to the 16-byte DICOM SH (Short String) limit."""

    return truncate_dicom_text(value, 16, default)


def truncate_lo(value: str | None, default: str = "") -> str:
    """Clamp ``value`` to the 64-byte DICOM LO (Long String) limit."""

    return truncate_dicom_text(value, 64, default)


def truncate_pn(value: str | None, default: str = "") -> str:
    """Clamp ``value`` to a conservative 64-byte DICOM PN value."""

    return truncate_dicom_text(value, 64, default)


def apply_synthetic_metadata(dataset, derivation_description: str) -> None:
    """Mark a generated SOP instance as synthetic and UTF-8 encoded."""

    dataset.SpecificCharacterSet = "ISO_IR 192"
    dataset.SyntheticData = "YES"
    dataset.DerivationDescription = truncate_dicom_text(
        derivation_description,
        1024,
        "Synthetic data generated by DICOMator from Blender geometry",
    )


def validate_numeric_array(
    value,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    """Return ``value`` as an array after shape, size, and finite checks."""

    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array")
    if array.size == 0 or any(int(length) <= 0 for length in array.shape):
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def resolve_positive_voxel_size(voxel_size) -> tuple[float, float, float]:
    """Return a finite positive three-component voxel size in metres."""

    if isinstance(voxel_size, (str, bytes)):
        raise ValueError("voxel_size must be a scalar or 3-component sequence")
    try:
        components = [float(component) for component in voxel_size]
    except TypeError:
        components = [float(voxel_size)] * 3
    if len(components) != 3:
        raise ValueError("voxel_size must be a scalar or 3-component sequence")
    if not all(np.isfinite(component) and component > 0.0 for component in components):
        raise ValueError("voxel_size components must be finite and greater than zero")
    return components[0], components[1], components[2]


#: Property names of the artifact toggles. Shared by the UI, the memory
#: estimate, and the export pipeline so they cannot drift apart when a new
#: artifact generator is added.
ARTIFACT_FLAGS = (
    "enable_noise",
    "enable_partial_volume",
    "enable_metal_artifacts",
    "enable_ring_artifacts",
    "enable_motion_artifact",
    "enable_poisson_noise",
    "enable_bias_field",
    "enable_geometric_distortion",
    "enable_gibbs_ringing",
)

#: Export guardrails. The export operator refuses grids beyond these unless
#: 'Allow Oversized Grids' is enabled, and the panels warn about them, so both
#: read the same numbers from here.
MAX_GRID_DIMENSION = 2000
MAX_TOTAL_VOXELS = 100_000_000
MAX_ESTIMATED_MEMORY_BYTES = 2 * 1024**3


def estimate_peak_memory_bytes(
    total_voxels: int,
    *,
    export_image_series: bool,
    export_drr: bool,
    export_rtdose: bool,
    artifacts_enabled: bool,
    gibbs_enabled: bool,
) -> int:
    """Conservatively estimate peak array memory for the selected pipeline."""

    total = max(0, int(total_voxels))
    image_bytes = 0
    if export_image_series or export_drr:
        image_bytes = 2
        if artifacts_enabled:
            image_bytes += 24
        if gibbs_enabled:
            image_bytes += 16
        if export_drr:
            image_bytes += 8
    # Voxel grid, clipped float32 dose, scaled float32 values, and uint32
    # frames can coexist briefly during RT Dose encoding.
    dose_bytes = 16 if export_rtdose else 0
    return total * max(image_bytes, dose_bytes, 2)


def estimate_peak_memory_bytes_for_props(total_voxels: int, props) -> int:
    """Return the peak-memory estimate for the outputs selected in ``props``."""

    return estimate_peak_memory_bytes(
        total_voxels,
        export_image_series=bool(getattr(props, "export_image_series", True)),
        export_drr=bool(getattr(props, "export_drr", False)),
        export_rtdose=bool(getattr(props, "export_rtdose", False)),
        artifacts_enabled=any(getattr(props, flag, False) for flag in ARTIFACT_FLAGS),
        gibbs_enabled=bool(getattr(props, "enable_gibbs_ringing", False)),
    )


def grid_limits_exceeded(
    width: int,
    height: int,
    depth: int,
    estimated_peak_bytes: int,
) -> bool:
    """Return True when a voxel grid is past the export guardrails."""

    dimensions = (int(width), int(height), int(depth))
    return (
        max(dimensions) > MAX_GRID_DIMENSION
        or dimensions[0] * dimensions[1] * dimensions[2] > MAX_TOTAL_VOXELS
        or int(estimated_peak_bytes) > MAX_ESTIMATED_MEMORY_BYTES
    )


def describe_grid_limits() -> str:
    """Return a human-readable summary of the export guardrails."""

    return (
        f"{MAX_GRID_DIMENSION:,} voxels per dimension, "
        f"{MAX_TOTAL_VOXELS:,} total, and "
        f"{MAX_ESTIMATED_MEMORY_BYTES / (1024**3):.0f} GiB estimated peak array memory"
    )


# ---------------------------------------------------------------------------
# RT DICOM SOP class UIDs
# ---------------------------------------------------------------------------

#: SOP Class UID for RT Structure Set (DICOM PS3.4 B.5)
RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"

#: SOP Class UID for RT Dose (DICOM PS3.4 B.5)
RTDOSE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.2"

#: SOP Class UID for RT Plan (DICOM PS3.4 B.5)
RTPLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"

# ---------------------------------------------------------------------------
# Per-object DICOM type items (used as EnumProperty items on bpy.types.Object)
# ---------------------------------------------------------------------------

DICOM_OBJECT_TYPE_ITEMS = [
    ("CT", "Image", "Contribute to CT/MR image series and DRR exports"),
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
]

PYDICOM_AVAILABLE = False
Dataset = None
FileDataset = None
generate_uid = None
pydicom = None
PYDICOM_IMPORT_ERROR = ""

#: True once an import attempt has run (success or failure). Callers such as
#: panel draw() invoke :func:`ensure_pydicom_available` on every UI redraw;
#: without this negative cache each redraw would re-glob the wheels directory
#: and retry the import when pydicom is missing.
_PYDICOM_IMPORT_ATTEMPTED = False


def ensure_pydicom_available(*, force_retry: bool = False) -> bool:
    """Import ``pydicom`` on demand and cache the resolved module globals.

    The result (including failure) is cached; pass ``force_retry=True`` to
    attempt the import again after the Python environment changes.
    """

    global PYDICOM_AVAILABLE
    global PYDICOM_IMPORT_ERROR
    global Dataset
    global FileDataset
    global generate_uid
    global pydicom
    global _PYDICOM_IMPORT_ATTEMPTED

    if PYDICOM_AVAILABLE and pydicom is not None and Dataset is not None and FileDataset is not None and generate_uid is not None:
        return True

    if _PYDICOM_IMPORT_ATTEMPTED and not force_retry:
        return False

    _PYDICOM_IMPORT_ATTEMPTED = True

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


__all__ = [
    "AIR_DENSITY",
    "DEFAULT_DENSITY",
    "MAX_HU_VALUE",
    "MIN_HU_VALUE",
    "MODALITY_CT",
    "MODALITY_MRI_T1",
    "MODALITY_MRI_T2",
    "MRI_MODALITIES",
    "IMAGING_MODALITY_ITEMS",
    "MATERIAL_INTENSITIES",
    "MATERIAL_ITEMS",
    "MR_SEQUENCE_PARAMETERS",
    "get_material_intensity",
    "apply_synthetic_metadata",
    "ARTIFACT_FLAGS",
    "MAX_GRID_DIMENSION",
    "MAX_TOTAL_VOXELS",
    "MAX_ESTIMATED_MEMORY_BYTES",
    "describe_grid_limits",
    "estimate_peak_memory_bytes",
    "estimate_peak_memory_bytes_for_props",
    "grid_limits_exceeded",
    "format_ds",
    "format_ds_sequence",
    "normalize_dicom_date",
    "resolve_positive_voxel_size",
    "sanitize_dicom_text",
    "truncate_dicom_text",
    "truncate_lo",
    "truncate_pn",
    "truncate_sh",
    "validate_numeric_array",
    "RTSTRUCT_SOP_CLASS",
    "RTDOSE_SOP_CLASS",
    "RTPLAN_SOP_CLASS",
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
