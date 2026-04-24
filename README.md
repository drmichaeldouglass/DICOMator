# DICOMator

Blender add-on that converts selected mesh objects into DICOM outputs for synthetic CT/MR image series, camera-based digitally reconstructed radiographs (DRRs), RT Dose grids, and RT Structure Sets. It voxelizes the active mesh selection directly into modality-appropriate intensities, supports single-phase or 4D acquisitions, and layers in synthetic artifacts tailored to the chosen modality for training or visualization workflows.

## Features

- **Per-object DICOM type and intensities**
  - Each selected mesh is tagged as **Image**, **RT Dose**, or **RT Structure** via the Objects panel
  - Image objects: set HU/intensity value or pick a tissue preset; overlapping meshes resolve by alphabetical name order
  - RT Dose objects: assign an absorbed dose value (Gy) per mesh; voxels within the mesh receive that dose
  - RT Structure objects: assign an ROI type (GTV, CTV, PTV, OAR, External, Other); contours are extracted at each CT slice plane
- **Tissue intensity presets**
  - Choose CT, T1 MR, or T2 MR modalities and assign tissue presets from a curated table
  - Selected presets automatically populate per-object intensities while still allowing manual overrides
- **Single-phase or 4D export**
  - Export the current frame or a range of frames (timeline or custom range)
  - One `SeriesInstanceUID` per phase; phases are written as separate series with temporal DICOM tags (`NumberOfTemporalPositions`, `TemporalPositionIndex`, `TemporalPositionIdentifier`)
  - Timeline advances during 4D export and a fixed padded bounding box keeps grids aligned between phases
- **Camera-based DRR generation**
  - Enable DRR output alongside, or instead of, the image series output
  - The DRR is generated from the active scene camera using a Beer-Lambert projection through the voxelized HU volume
  - Detector size follows the Blender render resolution with an optional DRR resolution scale
- **Voxelization control**
  - Independent lateral (XY) and axial (Z) voxel size in millimeters
  - Optional evaluation of modifiers/shape keys/armatures during voxelization
  - BVH-based +Z column fill for solid voxelization with consistent grid dimensions
- **Synthetic artifact suite**
  - CT modality exposes Gaussian noise, scanner point-spread partial volume, projection-like metal streaks, detector-channel rings, motion blur, and quantum noise
  - MRI modalities expose Gaussian noise, coil-shaped bias-field shading, and motion blur tuned for MR appearance
  - Artifact order matches the UI and adapts per modality so effects apply consistently
- **Patient and orientation metadata**
  - Patient Name, MRN (Patient ID), Sex, and Patient Position (HFS/FFS/HFP/FFP/HFDR/HFDL/FFDR/FFDL)
  - Customizable Series Description per export or phase
- **Export path handling and progress feedback**
  - Accepts Blender-relative paths starting with `//` (resolved relative to the `.blend` file or current working directory)
  - Defaults to `//DICOM_Export` so a new install starts with a portable export location instead of an OS-specific absolute path
  - Progress bar feedback for voxelization and slice writing
- **RT Dose export**
  - Mesh objects tagged as RT Dose are voxelized and written as a single multi-frame DICOM RT Dose file (`RTDoseStorage`)
  - Dose values (Gy) are encoded as uint32 scaled by a `DoseGridScaling` factor computed from the peak dose in the grid
  - Configurable `DoseType` (Physical / Effective) and `DoseSummationType` (Plan / Fraction / Beam)
  - RT Dose, image, DRR, and RT Structure exports share the same Study Instance UID and Frame of Reference UID when enabled together
- **RT Structure Set export**
  - Mesh objects tagged as RT Structure are sliced at each CT Z-plane using bmesh bisection
  - Closed planar contours are extracted from cut edges and written as a DICOM RT Structure Set (`RTStructureSetStorage`)
  - ROI display colour is read from the object's first material diffuse colour, or a clinical-style palette is used as a fallback
  - Supports ROI types GTV, CTV, PTV, OAR, External, and Other
- **Selection insights**
  - Live estimates of grid dimensions, voxel counts, and approximate memory usage before export

## Requirements

- Blender 5.1.1+ (Python 3.13 runtime)
- NumPy (bundled with Blender, used for grid operations)
- pydicom 3.0.1+ (required for DICOM export; vendored in `wheels/`; the add-on warns and disables export if missing)
- Optional helper wheels
  - Run `download_wheels.py` to download a Blender-targeted `pydicom` wheel into `./wheels` (defaults to Blender 5.1 / Python 3.13 tags, overrideable via environment variables)

## Blender 5.1.1 compatibility

- Targets the Blender 5.1.1 Python 3.13 runtime (VFX Platform 2026).
- The add-on has **no OpenVDB dependency**. Voxelization is performed with Blender mesh evaluation, `mathutils.bvhtree.BVHTree`, and NumPy arrays.
- NumPy 2.x compatibility is maintained by using `np.asarray(...)` rather than `np.array(..., copy=False)` throughout.
- Extension packaging advertises only the vendored `pydicom 3.0.1` (`py3-none-any`) wheel, which requires Python 3.10+ and is compatible with NumPy 2.x and Python 3.13.

## Installation

Blender 5.1.1 uses the **Extensions** workflow for add-ons.

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
   - The main **DICOMator** panel shows the selected object mix, output checkboxes, and export button.
   - **Objects** – For each selected mesh, choose its **DICOM Type**:
     - *Image*: assign HU/intensity values or pick modality-aware tissue presets. When meshes overlap, alphabetical ordering of object names decides the winning intensity (last name wins).
     - *RT Dose*: assign an absorbed dose in Gy. Voxels within the mesh receive that dose value when the RT Dose grid is built.
     - *RT Structure*: assign an ROI type (GTV, CTV, PTV, OAR, External, Other). The object's material diffuse colour is used as the ROI display colour in the structure set; a clinical-style palette is used if no material is assigned.
   - Enable the desired outputs in the main panel:
     - **Image** – writes CT or MR slices from Image meshes
     - **DRR** – writes a camera-based projection from Image meshes
     - **Dose** – writes RT Dose from RT Dose meshes
     - **Structures** – writes RT Structure Set from RT Structure meshes
   - **Export**
     - Configure **Lateral (mm)** and **Axial (mm)** voxel spacing
     - Toggle **Apply Modifiers/Deformations** to evaluate modifiers, armatures, and shape keys during voxelization
     - When DRR is enabled, set **DRR Resolution Scale** to scale the Blender render resolution used for the projection detector
     - When any RT Dose mesh is selected, dose settings appear for **Dose Type** (Physical / Effective) and **Dose Summation Type** (Plan / Fraction / Beam)
     - Choose an **Export Directory** (supports `//` relative paths and defaults to `//DICOM_Export`)
     - Toggle **Export 4D** to export multiple frames
       - Use the timeline range or set a custom `Start`/`End`/`Frame Step`
   - **Series** – Set the series description, patient name, MRN, sex, and patient position in one place.
   - **Estimate** – Inspect selection size, estimated grid resolution, voxel count, memory, and DRR detector dimensions.
   - **Artifacts** – Optional and collapsed by default for Image output:
     - *CT*: Gaussian noise, partial volume, metal streaks, rings, motion, and quantum noise
     - *MRI (T1/T2)*: Gaussian noise, coil bias-field shading, and motion
3. Click **Export DICOM**.
   - For single-phase exports the mesh selection is voxelized once and written directly in HU.
   - For 4D exports the timeline advances through the configured frame range, re-voxelizing each phase inside a fixed padded bounding box so every phase shares identical grid dimensions. Each phase receives its own Series Instance UID and the series description is suffixed with the phase number and percent completion.
   - When DRR is enabled, the voxelized HU volume is projected from the active camera into a single DICOM secondary-capture image per phase. CT presets are recommended because the DRR attenuation model assumes HU-like values.
   - When Dose or Structures are enabled, matching per-object meshes are exported alongside the image/DRR outputs in the same output directory, all sharing the same Study Instance UID and Frame of Reference UID.

Notes:
- During 4D export the timeline visibly advances; keep animation drivers and dependencies evaluated.
- Export filenames include the modality (e.g., `CT_Slice_####.dcm` or `MR_Slice_####.dcm`). 4D phases are prefixed with `Phase_###_` followed by the modality.
- When using relative (`//`) paths, save your `.blend` file so the path resolves predictably.

## Output details

- **Image Series output**
  - Modality: CT (`CTImageStorage`) or MR (`MRImageStorage`) depending on the selected imaging modality
  - Data type: int16 signed (direct HU/intensity values)
  - Window: CT exports default to Center 40 / Width 400; MR exports use Center 128 / Width 256
  - Geometry:
    - `PixelSpacing = [voxel_size_mm_y, voxel_size_mm_x]`
    - `SliceThickness = SpacingBetweenSlices = voxel_size_mm_z`
    - `ImageOrientationPatient = [1,0,0, 0,1,0]` (axial, aligned to world axes)
    - `ImagePositionPatient` derived from the padded world-space bounding box origin
- **RT Dose output**
  - Storage class: `RTDoseStorage` (SOP class 1.2.840.10008.5.1.4.1.1.481.2)
  - Data type: uint32 multi-frame image scaled by `DoseGridScaling` (Gy/count); maximum dose maps to the full uint32 range
  - Grid dimensions and spatial coordinates match the CT grid exactly, ensuring voxel-to-voxel correspondence
  - `DoseType`, `DoseSummationType`, and `DoseUnits = GY` are set from the RT Dose Settings panel
  - Shares `StudyInstanceUID` and `FrameOfReferenceUID` with co-exported image, DRR, and structure outputs
- **RT Structure Set output**
  - Storage class: `RTStructureSetStorage` (SOP class 1.2.840.10008.5.1.4.1.1.481.3)
  - Contours generated by bisecting each structure mesh at every CT Z-plane using bmesh operations
  - Only closed planar loops with three or more points are written; open or degenerate edge chains are discarded
  - Each ROI's colour is taken from the object's material diffuse colour (0–255 RGB); a clinical-style palette provides fallback colours
  - References the co-exported image series when Image output is enabled
- **DRR output**
  - Storage class: Secondary Capture (`SecondaryCaptureImageStorage`) with `ImageType = DERIVED\\PRIMARY\\DRR`
  - Data type: uint16 monochrome projection image
  - Geometry:
    - `PixelSpacing` follows the active camera detector plane dimensions divided by detector pixels
    - `ImageOrientationPatient` follows the active camera detector row/column axes
    - `ImagePositionPatient` is written from the detector plane top-left corner in world coordinates
- Temporal DICOM tags (4D only):
  - `NumberOfTemporalPositions` (total phases)
  - `TemporalPositionIndex` (1-based phase order)
  - `TemporalPositionIdentifier` (1-based phase order)

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
- DRR generation projects the voxelized volume rather than the original triangle mesh, so image sharpness depends on the chosen voxel spacing.
- Modifier/armature evaluation is optional but increases memory/time usage; complex rigs may still require baking.
- Output orientation is fixed to axial slices aligned with Blender world axes.
- DRR export requires an active scene camera and uses image-type meshes without the CT/MR artifact stack.
- RT Structure Set contour extraction is performed per-slice using planar bisection; very thin or highly curved structures may produce incomplete contours at coarse voxel spacings.
- Each mesh contributes to exactly one DICOM object type; use duplicate meshes if the same geometry should be exported as multiple object types.
- Only the modality-specific artifacts listed above are available; additional acquisition effects are not modeled in this release.

## Troubleshooting

- **“pydicom library not available”**
  - Install `pydicom` in Blender’s Python or use the helper script to fetch wheels, then restart Blender.
- **“Grid too large”**
  - Increase voxel spacing (mm), reduce padding/selection size, or limit the frame range.
- **“Set an active scene camera before exporting a DRR”**
  - Assign a camera to the scene (`Scene Properties → Camera`) or make a camera active in the 3D View before DRR export.
- **“Output directory is not writable”**
  - Choose a folder with write permissions; blank export paths are rejected, and for `//` paths, save your `.blend` so the relative path resolves.
- **Artifacts look too strong/weak**
  - Adjust intensity/severity controls or disable individual artifact toggles to isolate effects.
- **RT Structure contours missing or incomplete**
  - Ensure structure meshes are closed (manifold) and intersect the CT grid Z-planes. Increase axial resolution (decrease voxel spacing) to capture thin structures.
- **RT Dose grid is all zeros**
  - Verify that the mesh objects intended as dose volumes have their DICOM Type set to *RT Dose* and that a non-zero Dose (Gy) value is assigned in the Objects panel.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
