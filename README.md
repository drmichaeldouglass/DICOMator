# DICOMator

Blender add-on that converts selected mesh objects into a DICOM CT image series. It voxelizes the active mesh selection directly into Hounsfield Units, supports single-phase or 4D acquisitions, and can layer in synthetic CT artifacts for training or visualization workflows.

## Features

- **Per-object Hounsfield Units (HU)**
  - Set HU on each mesh (Object property `HU` via the Per-Object HU panel)
  - Overlapping meshes resolve deterministically by mesh name; alphabetically last meshes win when voxels coincide
- **Tissue intensity presets**
  - Choose CT, T1 MR, or T2 MR modalities and assign tissue presets from a curated table
  - Selected presets automatically populate per-object intensities while still allowing manual overrides
- **Single-phase or 4D export**
  - Export the current frame or a range of frames (timeline or custom range)
  - One `SeriesInstanceUID` per phase; phases are written as separate series with temporal DICOM tags (`NumberOfTemporalPositions`, `TemporalPositionIndex`, `TemporalPositionIdentifier`)
  - Timeline advances during 4D export and a fixed padded bounding box keeps grids aligned between phases
- **Voxelization control**
  - Independent lateral (XY) and axial (Z) voxel size in millimeters
  - Optional evaluation of modifiers/shape keys/armatures during voxelization
  - BVH-based +Z column fill for solid voxelization with consistent grid dimensions
- **Synthetic CT artifact suite**
  - Partial volume blur, metal streaks, ring artifacts, motion blur, Gaussian noise, and Poisson noise can be combined
  - Artifact order matches the UI: partial volume → metal streaks → rings → motion → Gaussian → Poisson
- **Patient and orientation metadata**
  - Patient Name, MRN (Patient ID), Sex, and Patient Position (HFS/FFS/HFP/FFP/HFDR/HFDL/FFDR/FFDL)
  - Customizable Series Description per export or phase
- **Export path handling and progress feedback**
  - Accepts Blender-relative paths starting with `//` (resolved relative to the `.blend` file or current working directory)
  - Progress bar feedback for voxelization and slice writing
- **Selection insights**
  - Live estimates of grid dimensions, voxel counts, and approximate memory usage before export

## Requirements

- Blender 4.2+
- NumPy (bundled with Blender, used for grid operations)
- pydicom (required for DICOM export; the add-on warns and disables export if missing)
- Optional helper wheels
  - Run `download_wheels.py` to download Windows-compatible wheels (pydicom, scikit-image, SciPy) into `./wheels`

## Installation

1. Zip the `dicomator` folder or install the add-on folder directly from Blender:
   - Blender → **Edit → Preferences → Add-ons → Install…**
   - Select the packaged zip or folder, then enable **“DICOMator”**.
2. Ensure `pydicom` is available. Install it into Blender’s Python environment or use the provided helper script (`python download_wheels.py`) and point Blender to the downloaded wheels.

## Usage

1. Select one or more mesh objects in the 3D Viewport.
2. In **Sidebar → DICOMator**, configure the panels:
   - **Selection Info** – Inspect selection size, estimated grid resolution, voxel count, and memory. Guardrails warn when exceeding 2,000 voxels per axis or 100M total voxels.
  - **Per-Object HU** – Assign HU values or pick modality-aware tissue presets for each selected mesh. When meshes overlap, alphabetical ordering of object names decides the winning intensity (last name wins).
   - **Patient Information** – Set patient name, MRN, and sex.
   - **Image Orientation** – Choose the Patient Position tag applied to the DICOM slices.
   - **Export Settings**
     - Configure **Lateral (mm)** and **Axial (mm)** voxel spacing
     - Toggle **Apply Modifiers/Deformations** to evaluate modifiers, armatures, and shape keys during voxelization
     - Choose an **Export Directory** (supports `//` relative paths)
     - Toggle **Export 4D** to export multiple frames
       - Use the timeline range or set a custom `Start`/`End`/`Frame Step`
     - Enter a **Series Description** (used directly or extended per phase)
   - **CT Artifacts** – Enable and tune optional artifact simulations:
     - *Gaussian Noise*: zero-mean HU noise with configurable standard deviation
     - *Partial Volume Blur*: volumetric smoothing with kernel size, iterations, and blend control
     - *Metal Streaks*: streak artifacts originating from voxels above a HU threshold (intensity, streak count, falloff)
     - *Ring Artifacts*: concentric banding with adjustable amplitude, ring count, and jitter
     - *Motion Blur*: in-plane motion blur/ghosting with odd-length kernel, severity, and axis choice
     - *Poisson Noise*: photon-count style noise governed by a photon scale factor
3. Click **Export to DICOM**.
   - For single-phase exports the mesh selection is voxelized once and written directly in HU.
   - For 4D exports the timeline advances through the configured frame range, re-voxelizing each phase inside a fixed padded bounding box so every phase shares identical grid dimensions. Each phase receives its own Series Instance UID and the series description is suffixed with the phase number and percent completion.

Notes:
- During 4D export the timeline visibly advances; keep animation drivers and dependencies evaluated.
- Export filenames default to `CT_Slice_####.dcm`. 4D phases are prefixed with `Phase_###_`.
- When using relative (`//`) paths, save your `.blend` file so the path resolves predictably.

## Output details

- Modality: CT (CT Image Storage)
- Data type: int16 signed (direct HU values)
- Window: Center 40, Width 400
- Geometry:
  - `PixelSpacing = [voxel_size_mm_y, voxel_size_mm_x]`
  - `SliceThickness = SpacingBetweenSlices = voxel_size_mm_z`
  - `ImageOrientationPatient = [1,0,0, 0,1,0]` (axial, aligned to world axes)
  - `ImagePositionPatient` derived from the padded world-space bounding box origin
- Temporal DICOM tags (4D only):
  - `NumberOfTemporalPositions` (total phases)
  - `TemporalPositionIndex` (timeline frame number)
  - `TemporalPositionIdentifier` (1-based phase index)

![skull_multi](https://github.com/user-attachments/assets/b1c62567-4189-4a66-812f-005b57629184)
![skull_dose_lat](https://github.com/user-attachments/assets/eca22ede-4a6f-47ca-a82c-e53dccb0649d)
![Lung_geometry](https://github.com/user-attachments/assets/8eb7a3ce-fbaf-4d7d-b70d-33e7b808e0fd)
![Lung](https://github.com/user-attachments/assets/77e204bd-2a70-46bb-af8f-c3327ef7eb8f)

## Performance and limits

- Guardrails prevent extremely large grids:
  - Per-dimension limit: 2,000 voxels
  - Total voxels limit: 100,000,000 (≈200 MB int16)
- Tips:
  - Increase voxel spacing (mm) to reduce memory/time requirements
  - Reduce the number of selected meshes or animation frames
  - Disable modifier evaluation if you only need undeformed geometry
  - Narrow the frame range or increase frame step for 4D exports
  - Disable artifact generators when testing baseline exports

## Known limitations

- Voxelization is axis-aligned and uses +Z column fills; only mesh geometry is sampled (materials/textures are ignored).
- Modifier/armature evaluation is optional but increases memory/time usage; complex rigs may still require baking.
- Output orientation is fixed to axial slices aligned with Blender world axes.
- Only synthetic artifacts listed above are available; additional acquisition effects are not modeled in this release.

## Troubleshooting

- **“pydicom library not available”**
  - Install `pydicom` in Blender’s Python or use the helper script to fetch wheels, then restart Blender.
- **“Grid too large”**
  - Increase voxel spacing (mm), reduce padding/selection size, or limit the frame range.
- **“Output directory is not writable”**
  - Choose a folder with write permissions; for `//` paths, save your `.blend` so the relative path resolves.
- **Artifacts look too strong/weak**
  - Adjust intensity/severity controls or disable individual artifact toggles to isolate effects.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
