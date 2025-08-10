# DICOMator

Blender add-on that converts selected mesh objects into a DICOM CT image series. Supports single-frame and 4D (multi-phase) exports, per-object Hounsfield units, and optional Gaussian noise.

## Features

- Per-object Hounsfield Units (HU)
  - Set HU on each mesh (Object property “HU” via the Per-Object HU panel)
  - Overlapping meshes resolve by taking the maximum HU
- Single-frame or 4D export
  - Export the current frame or a range of frames (timeline or custom range)
  - One SeriesInstanceUID per phase; phases are written as separate series
  - Temporal tags included:
    - NumberOfTemporalPositions = total phases
    - TemporalPositionIndex = timeline frame number
    - TemporalPositionIdentifier = 1-based phase index
- Voxelization control
  - Grid resolution in millimeters (voxel size)
  - Stable grid dimensions across phases using a fixed padded bounding box
  - BVH-based +Z column fill for fast solid voxelization
- Patient and orientation metadata
  - Patient Name, MRN (Patient ID), Sex
  - Patient Position (HFS/FFS/HFP/FFP/HFDR/HFDL/FFDR/FFDL)
- Optional Gaussian noise
  - Enable/disable noise
  - Noise standard deviation in HU
- Export path handling
  - Accepts Blender-relative paths starting with // (resolved relative to the .blend or CWD)

## Requirements

- Blender 4.2+
- NumPy (used for grid operations)
- pydicom (required for DICOM export; the add-on warns and disables export if missing)
- Optional: use the provided helper to fetch wheels
  - Run download_wheels.py to download wheels into ./wheels

## Installation

1. Zip the dicomator folder or install the add-on folder directly from Blender:
   - Blender → Edit → Preferences → Add-ons → Install…
   - Select the packaged zip or folder, then enable “DICOMator”.

2. Ensure pydicom is available. If not, install pydicom or use the wheels helper script.

## Usage

1. Select one or more mesh objects in the 3D Viewport.
2. In Sidebar → DICOMator:
   - Selection Info: review object size and estimated grid/memory.
   - Per-Object HU: set each selected object’s HU (default 0 HU).
   - Patient Information: set name, MRN, and sex.
   - Image Orientation: choose Patient Position.
   - Export Settings:
     - Grid Resolution (mm)
     - Export Directory (// paths resolve to the blend folder or current working directory)
     - Series Description
     - Gaussian Noise: enable and set Noise Std. Dev. (HU) if desired
     - 4D export:
       - Enable Export 4D
       - Use timeline range or specify Start/End and Frame Step
3. Click “Export to DICOM”.

Notes:
- During 4D export the timeline visibly advances. The grid size is fixed across phases for proper series alignment.
- Each phase is exported to its own DICOM series and filenames are prefixed with phase, e.g. Phase_001_CT_Slice_0001.dcm.

## Output details

- Modality: CT (CT Image Storage)
- Data type: int16 signed
- Window: Center 40, Width 400
- Geometry:
  - PixelSpacing = [voxel_size_mm, voxel_size_mm]
  - SliceThickness = SpacingBetweenSlices = voxel_size_mm
  - ImageOrientationPatient = [1,0,0, 0,1,0] (axial identity)
  - ImagePositionPatient derived from the fixed world-space bounding box origin
- Temporal DICOM tags (4D):
  - NumberOfTemporalPositions (total phases)
  - TemporalPositionIndex (timeline frame)
  - TemporalPositionIdentifier (1-based phase)

![skull_multi](https://github.com/user-attachments/assets/b1c62567-4189-4a66-812f-005b57629184)
![skull_dose_lat](https://github.com/user-attachments/assets/eca22ede-4a6f-47ca-a82c-e53dccb0649d)
![Lung_geometry](https://github.com/user-attachments/assets/8eb7a3ce-fbaf-4d7d-b70d-33e7b808e0fd)
![Lung](https://github.com/user-attachments/assets/77e204bd-2a70-46bb-af8f-c3327ef7eb8f)


## Performance and limits

- Guardrails prevent extremely large grids:
  - Per-dimension limit: 2000 voxels
  - Total voxels limit: 100,000,000 (≈200 MB int16)
- Tips:
  - Increase Grid Resolution (mm) to reduce memory/time
  - Reduce the number of selected meshes
  - Narrow the frame range or increase frame step for 4D

## Known limitations

- Geometry evaluation:
  - The exporter builds BVHs from object.data transformed by matrix_world
  - Modifiers/armatures are not evaluated during export
  - Apply modifiers or convert to mesh if you need deformed geometry baked into the export
- Only Gaussian noise is implemented (other artifact simulations are not present in this version)

## Troubleshooting

- “pydicom library not available”
  - Install pydicom in Blender’s Python or use the helper script to fetch wheels
- “Grid too large”
  - Increase Grid Resolution (mm) or reduce the selection size
- “Output directory is not writable”
  - Choose a folder with write permissions; for // paths, save your .blend so // resolves to a known directory

## License

This project is released under the MIT License. See LICENSE for details.

