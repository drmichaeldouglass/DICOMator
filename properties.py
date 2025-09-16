"""Property group definitions for the DICOMator add-on."""
from __future__ import annotations

import bpy


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


__all__ = ["DICOMatorProperties"]
