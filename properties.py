"""Property group definitions for the DICOMator add-on."""
from __future__ import annotations

import bpy

from .constants import IMAGING_MODALITY_ITEMS, get_material_intensity


def apply_material_intensity(obj: bpy.types.Object, modality: str) -> None:
    """Assign the intensity for ``obj`` based on its material and modality."""

    material_key = getattr(obj, "dicomator_material", "CUSTOM")
    if material_key == "CUSTOM":
        return

    value = get_material_intensity(material_key, modality)
    if value is not None:
        obj.dicomator_hu = float(value)


def update_imaging_modality(self, context: bpy.types.Context) -> None:
    """Refresh material-derived intensities when the modality changes."""

    if context is None or context.scene is None:
        return

    for obj in context.scene.objects:
        if obj.type != 'MESH':
            continue
        apply_material_intensity(obj, self.imaging_modality)


def update_object_material(self, context: bpy.types.Context) -> None:
    """Update an object's intensity when its material preset changes."""

    if context is None or context.scene is None:
        return

    props = getattr(context.scene, "dicomator_props", None)
    if props is None:
        return

    modality = getattr(props, "imaging_modality", IMAGING_MODALITY_ITEMS[0][0])
    apply_material_intensity(self, modality)


class DICOMatorProperties(bpy.types.PropertyGroup):
    """Properties exposed in the DICOMator UI."""

    patient_name: bpy.props.StringProperty(
        name="Patient Name",
        description="Name of the patient",
        default="Anonymous",
    )
    patient_id: bpy.props.StringProperty(
        name="MRN",
        description="Medical Record Number (Patient ID)",
        default="12345678",
    )
    patient_sex: bpy.props.EnumProperty(
        name="Patient Sex",
        description="Patient sex",
        items=[
            ('M', 'Male', 'Male'),
            ('F', 'Female', 'Female'),
            ('O', 'Other', 'Other'),
        ],
        default='M',
    )
    patient_position: bpy.props.EnumProperty(
        name="Patient Position",
        description="DICOM Patient Position (image orientation context)",
        items=[
            ('HFS', 'Head First Supine', 'Head First Supine'),
            ('FFS', 'Feet First Supine', 'Feet First Supine'),
            ('HFP', 'Head First Prone', 'Head First Prone'),
            ('FFP', 'Feet First Prone', 'Feet First Prone'),
            ('HFDR', 'Head First Decubitus Right', 'Head First Decubitus Right'),
            ('HFDL', 'Head First Decubitus Left', 'Head First Decubitus Left'),
            ('FFDR', 'Feet First Decubitus Right', 'Feet First Decubitus Right'),
            ('FFDL', 'Feet First Decubitus Left', 'Feet First Decubitus Left'),
        ],
        default='HFS',
    )
    imaging_modality: bpy.props.EnumProperty(
        name="Imaging Modality",
        description="Select the imaging modality used for material presets",
        items=IMAGING_MODALITY_ITEMS,
        default=IMAGING_MODALITY_ITEMS[0][0],
        update=update_imaging_modality,
    )
    export_4d: bpy.props.BoolProperty(
        name="Export 4D (use animated frames)",
        description="Export each selected animation frame as a 4D CT phase",
        default=False,
    )
    use_timeline_range: bpy.props.BoolProperty(
        name="Use Timeline Range",
        description="Use the scene's timeline start/end as the frame range",
        default=True,
    )
    frame_start: bpy.props.IntProperty(
        name="Start Frame",
        description="First frame to export (when not using timeline range)",
        default=1,
        min=1,
    )
    frame_end: bpy.props.IntProperty(
        name="End Frame",
        description="Last frame to export (when not using timeline range)",
        default=250,
        min=1,
    )
    frame_step: bpy.props.IntProperty(
        name="Frame Step",
        description="Step between frames to export",
        default=1,
        min=1,
    )
    lateral_resolution_mm: bpy.props.FloatProperty(
        name="Lateral Resolution (mm)",
        description="Voxel size in X/Y (mm)",
        default=2.0,
        min=0.1,
        max=10.0,
        step=10,
        precision=2,
    )
    axial_resolution_mm: bpy.props.FloatProperty(
        name="Axial Resolution (mm)",
        description="Voxel size in Z (mm)",
        default=2.0,
        min=0.1,
        max=10.0,
        step=10,
        precision=2,
    )
    apply_modifiers: bpy.props.BoolProperty(
        name="Apply Modifiers/Deformations",
        description="Evaluate modifiers/shape keys/armatures/lattices when voxelizing",
        default=True,
    )
    export_directory: bpy.props.StringProperty(
        name="Export Directory",
        description="Directory to save DICOM files",
        subtype='DIR_PATH',
        default="C:\\Users\\Public\\DICOM_Export",
    )
    series_description: bpy.props.StringProperty(
        name="Series Description",
        description="Description for the DICOM series",
        default="CT Series from DICOMator",
    )
    enable_noise: bpy.props.BoolProperty(
        name="Add Gaussian Noise",
        description="Add zero-mean Gaussian noise to exported HU images",
        default=False,
    )
    noise_std_dev_hu: bpy.props.FloatProperty(
        name="Noise Std. Dev. (HU)",
        description="Standard deviation of Gaussian noise in Hounsfield Units",
        default=20.0,
        min=0.0,
        soft_max=500.0,
        step=10,
        precision=1,
    )
    enable_bias_field: bpy.props.BoolProperty(
        name="Add Bias Field Shading",
        description="Apply low-frequency coil shading variations (MRI)",
        default=False,
    )
    bias_field_strength: bpy.props.FloatProperty(
        name="Bias Strength",
        description="Fractional amplitude of the multiplicative bias field",
        default=0.25,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    bias_field_scale: bpy.props.FloatProperty(
        name="Bias Scale",
        description="Relative smoothing window controlling how quickly the bias varies",
        default=0.3,
        min=0.05,
        max=1.0,
        precision=2,
    )
    enable_partial_volume: bpy.props.BoolProperty(
        name="Add Partial Volume Blur",
        description="Blend materials at boundaries using a volumetric blur",
        default=False,
    )
    partial_volume_kernel: bpy.props.IntProperty(
        name="Kernel Size",
        description="Size of the averaging kernel (odd integer)",
        default=3,
        min=1,
        soft_max=11,
    )
    partial_volume_iterations: bpy.props.IntProperty(
        name="Iterations",
        description="Number of smoothing passes to apply",
        default=1,
        min=1,
        soft_max=5,
    )
    partial_volume_mix: bpy.props.FloatProperty(
        name="Blend",
        description="Blend factor between blurred and original data",
        default=1.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    enable_metal_artifacts: bpy.props.BoolProperty(
        name="Add Metal Streaks",
        description="Simulate streak artifacts originating from dense materials",
        default=False,
    )
    metal_intensity: bpy.props.FloatProperty(
        name="Streak Intensity (HU)",
        description="Base amplitude in HU for generated streaks",
        default=400.0,
        min=0.0,
        soft_max=2000.0,
        step=10,
    )
    metal_density_threshold: bpy.props.FloatProperty(
        name="Metal Threshold (HU)",
        description="HU value above which voxels count as metal",
        default=2000.0,
        min=0.0,
        soft_max=4000.0,
        step=10,
    )
    metal_num_streaks: bpy.props.IntProperty(
        name="Streak Count",
        description="Number of streaks per slice (0 = automatic)",
        default=10,
        min=0,
        soft_max=32,
    )
    metal_falloff: bpy.props.FloatProperty(
        name="Falloff",
        description="Spatial decay of streaks away from the streak axis",
        default=6.0,
        min=0.1,
        soft_max=12.0,
        step=10,
        precision=2,
    )
    enable_ring_artifacts: bpy.props.BoolProperty(
        name="Add Ring Artifacts",
        description="Introduce concentric banding similar to detector gain errors",
        default=False,
    )
    ring_intensity: bpy.props.FloatProperty(
        name="Ring Intensity (HU)",
        description="Amplitude of the generated rings in HU",
        default=80.0,
        min=0.0,
        soft_max=500.0,
        step=10,
    )
    ring_radius: bpy.props.FloatProperty(
        name="Ring Radius (rel)",
        description="Relative radius of the ring (0=center, 1=edge). Set to 0.0-1.0 or leave default to choose randomly",
        default=0.5,
        min=0.0,
        max=1.0,
        precision=3,
    )
    ring_thickness: bpy.props.FloatProperty(
        name="Ring Thickness (rel)",
        description="Relative radial thickness of the ring (typical ~0.01-0.05)",
        default=0.02,
        min=0.0,
        soft_max=0.1,
        precision=3,
    )
    ring_jitter: bpy.props.FloatProperty(
        name="Jitter",
        description="Noise mixed with rings to avoid smooth bands",
        default=0.02,
        min=0.0,
        soft_max=0.2,
        precision=3,
    )
    enable_motion_artifact: bpy.props.BoolProperty(
        name="Add Motion Blur",
        description="Simulate in-plane patient motion during acquisition",
        default=False,
    )
    motion_blur_size: bpy.props.IntProperty(
        name="Blur Length",
        description="Length of the motion blur kernel (odd integer)",
        default=9,
        min=1,
        soft_max=21,
    )
    motion_severity: bpy.props.FloatProperty(
        name="Severity",
        description="Blend factor between original and blurred slice",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    motion_axis: bpy.props.EnumProperty(
        name="Blur Axis",
        description="Direction of motion blur within each slice",
        items=[
            ('X', 'X Axis', 'Blur along the X axis (left/right)'),
            ('Y', 'Y Axis', 'Blur along the Y axis (anterior/posterior)'),
        ],
        default='X',
    )
    enable_poisson_noise: bpy.props.BoolProperty(
        name="Add Poisson Noise",
        description="Approximate photon counting noise via a Poisson process",
        default=False,
    )
    poisson_scale: bpy.props.FloatProperty(
        name="Photon Scale",
        description="Higher values reduce noise by increasing simulated counts",
        default=150.0,
        min=1.0,
        soft_max=1000.0,
        step=10,
    )


__all__ = [
    "DICOMatorProperties",
    "apply_material_intensity",
    "update_imaging_modality",
    "update_object_material",
]
