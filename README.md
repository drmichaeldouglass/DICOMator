# DICOMator

Blender add-on that converts selected mesh objects into DICOM outputs for either synthetic CT/MR image series or camera-based digital reconstructed radiographs (DRRs). It voxelizes the active mesh selection directly into modality-appropriate intensities, supports single-phase or 4D acquisitions, and layers in synthetic artifacts tailored to the chosen modality for training or visualization workflows.

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
- **Camera-based DRR generation**
  - Switch the reconstruction mode from synthetic volume export to DRR
  - The DRR is generated from the active scene camera using a Beer-Lambert projection through the voxelized HU volume
  - Detector size follows the Blender render resolution with an optional DRR resolution scale
- **Voxelization control**
  - Independent lateral (XY) and axial (Z) voxel size in millimeters
  - Optional evaluation of modifiers/shape keys/armatures during voxelization
  - BVH-based +Z column fill for solid voxelization with consistent grid dimensions
- **Synthetic artifact suite**
  - CT modality exposes partial volume blur, metal streaks, ring artifacts, motion blur, Gaussian noise, and Poisson noise
  - MRI modalities expose Gaussian noise, coil bias-field shading, and motion blur tuned for MR appearance
  - Artifact order matches the UI and adapts per modality so effects apply consistently
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
  - Run `download_wheels.py` to download a Blender-targeted `pydicom` wheel into `./wheels` (defaults to Blender 5.1 / Python 3.13 tags, overrideable via environment variables)

## Blender 5.1 compatibility review

- Reviewed against the Blender 5.1 Python API changes listed in the release notes shared in this issue. The add-on does **not** call any of the renamed VSE strip properties, removed brush stroke flags, `sculpt.sample_color`, deprecated `UILayout.template_list(columns=...)`, GPU framebuffer read helpers, or the new exit/cachedir APIs, so no Blender 5.1 API breakage was found in those areas.
- The add-on has **no OpenVDB dependency**. Voxelization is performed with Blender mesh evaluation, `mathutils.bvhtree.BVHTree`, and NumPy arrays, so the OpenVDB 13.x update does not affect the current code path.
- NumPy 2.x compatibility was tightened by replacing the direct `np.array(..., copy=False)` conversion used during DICOM export with `np.asarray(...)`, which avoids the stricter copy semantics introduced in NumPy 2.0+.
- Extension packaging was cleaned up so the manifest only advertises the vendored `pydicom` wheel that the add-on actually uses. The previous manifest listed SciPy and scikit-image wheels that are not required by the codebase and were not present in the repository.
- The repository now vendors `pydicom 3.0.1` (`py3-none-any`), which requires Python 3.10+ and is compatible with NumPy 2.x and Python 3.13, matching Blender 5.1's runtime.

## Installation

Blender 4.2+ uses the **Extensions** workflow for add-ons.

1. Create an extension zip from this repository root (the zip must include `blender_manifest.toml` at the top level).
2. In Blender, open **Edit → Preferences → Extensions**.
3. Open the Extensions menu (top-right caret) and choose **Install from Disk...**.
4. Select the zip file and install.
5. Enable **DICOMator** if it is not enabled automatically.

Development fallback (unpacked source):

1. Copy or symlink this folder into your Blender user scripts add-ons directory.
2. In Blender, open **Edit → Preferences → Add-ons**, search for **DICOMator**, and enable it.

Dependency note:

- Ensure `pydicom` is available in Blender's Python environment.

## Usage

1. Select one or more mesh objects in the 3D Viewport.
2. In **Sidebar → DICOMator**, configure the panels:
   - **Selection Info** – Inspect selection size, estimated grid resolution, voxel count, and memory. Guardrails warn when exceeding 2,000 voxels per axis or 100M total voxels.
     - When DRR mode is active, the panel also shows the active camera and estimated detector pixel dimensions.
   - **Per-Object HU** – Assign HU values or pick modality-aware tissue presets for each selected mesh. When meshes overlap, alphabetical ordering of object names decides the winning intensity (last name wins).
   - **Patient Information** – Set patient name, MRN, and sex.
   - **Image Orientation** – Choose the Patient Position tag applied to the DICOM slices.
   - **Export Settings**
     - Choose **Reconstruction**:
       - **Synthetic Volume** – writes CT or MR slices according to the selected imaging modality
       - **DRR** – writes a camera-based projection image from the active scene camera
     - Configure **Lateral (mm)** and **Axial (mm)** voxel spacing
     - Toggle **Apply Modifiers/Deformations** to evaluate modifiers, armatures, and shape keys during voxelization
     - In DRR mode, set **DRR Resolution Scale** to scale the Blender render resolution used for the projection detector
     - Choose an **Export Directory** (supports `//` relative paths)
     - Toggle **Export 4D** to export multiple frames
       - Use the timeline range or set a custom `Start`/`End`/`Frame Step`
     - Enter a **Series Description** (used directly or extended per phase)
   - **Artifact Controls** – Available for synthetic volume export only. The panel title changes with the modality:
     - *CT*: Gaussian noise, partial volume blur, metal streaks, ring artifacts, motion blur, and Poisson noise
     - *MRI (T1/T2)*: Gaussian noise (intensity-scaled), low-frequency coil bias-field shading, and motion blur
3. Click **Export to DICOM** or **Export DRR**.
   - For single-phase exports the mesh selection is voxelized once and written directly in HU.
   - For 4D exports the timeline advances through the configured frame range, re-voxelizing each phase inside a fixed padded bounding box so every phase shares identical grid dimensions. Each phase receives its own Series Instance UID and the series description is suffixed with the phase number and percent completion.
   - In DRR mode, the voxelized HU volume is projected from the active camera into a single DICOM secondary-capture image per phase. If MRI presets are selected, the export still works, but CT presets are recommended because the DRR attenuation model assumes HU-like values.

Notes:
- During 4D export the timeline visibly advances; keep animation drivers and dependencies evaluated.
- Export filenames include the modality (e.g., `CT_Slice_####.dcm` or `MR_Slice_####.dcm`). 4D phases are prefixed with `Phase_###_` followed by the modality.
- When using relative (`//`) paths, save your `.blend` file so the path resolves predictably.

## Output details

- **Synthetic Volume mode**
  - Modality: CT (`CTImageStorage`) or MR (`MRImageStorage`) depending on the selected imaging modality
  - Data type: int16 signed (direct HU/intensity values)
  - Window: CT exports default to Center 40 / Width 400; MR exports use Center 128 / Width 256
  - Geometry:
    - `PixelSpacing = [voxel_size_mm_y, voxel_size_mm_x]`
    - `SliceThickness = SpacingBetweenSlices = voxel_size_mm_z`
    - `ImageOrientationPatient = [1,0,0, 0,1,0]` (axial, aligned to world axes)
    - `ImagePositionPatient` derived from the padded world-space bounding box origin
- **DRR mode**
  - Storage class: Secondary Capture (`SecondaryCaptureImageStorage`) with `ImageType = DERIVED\\PRIMARY\\DRR`
  - Data type: uint16 monochrome projection image
  - Geometry:
    - `PixelSpacing` follows the active camera detector plane dimensions divided by detector pixels
    - `ImageOrientationPatient` follows the active camera detector row/column axes
    - `ImagePositionPatient` is written from the detector plane top-left corner in world coordinates
- Temporal DICOM tags (4D only):
  - `NumberOfTemporalPositions` (total phases)
  - `TemporalPositionIndex` (timeline frame number)
  - `TemporalPositionIdentifier` (1-based phase index)

<img src="https://github.com/user-attachments/assets/b1c62567-4189-4a66-812f-005b57629184" alt="skull_multi" width="420" />
<img src="https://github.com/user-attachments/assets/eca22ede-4a6f-47ca-a82c-e53dccb0649d" alt="skull_dose_lat" width="420" />
<img src="https://github.com/user-attachments/assets/8eb7a3ce-fbaf-4d7d-b70d-33e7b808e0fd" alt="Lung_geometry" width="420" />
<img src="https://github.com/user-attachments/assets/77e204bd-2a70-46bb-af8f-c3327ef7eb8f" alt="Lung" width="420" />
<img width="359" height="362" alt="SuzanneXRay" src="https://github.com/user-attachments/assets/2951918f-773b-4505-88cc-4086dfd64b2c" width="420"/>

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
- DRR generation currently projects the voxelized volume rather than the original triangle mesh, so image sharpness depends on the chosen voxel spacing.
- Modifier/armature evaluation is optional but increases memory/time usage; complex rigs may still require baking.
- Output orientation is fixed to axial slices aligned with Blender world axes.
- DRR export requires an active scene camera and currently uses the existing HU grid without the synthetic CT/MR artifact stack.
- Only the modality-specific artifacts listed above are available; additional acquisition effects are not modeled in this release.

## Troubleshooting

- **“pydicom library not available”**
  - Install `pydicom` in Blender’s Python or use the helper script to fetch wheels, then restart Blender.
- **“Grid too large”**
  - Increase voxel spacing (mm), reduce padding/selection size, or limit the frame range.
- **“Set an active scene camera before exporting a DRR”**
  - Assign a camera to the scene (`Scene Properties → Camera`) or make a camera active in the 3D View before DRR export.
- **“Output directory is not writable”**
  - Choose a folder with write permissions; for `//` paths, save your `.blend` so the relative path resolves.
- **Artifacts look too strong/weak**
  - Adjust intensity/severity controls or disable individual artifact toggles to isolate effects.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
