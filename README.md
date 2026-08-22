# DICOMator

<p align="center">
  <img src="docs/assets/dicomator-logo.png" alt="DICOMator voxel cube logo" width="256" />
</p>

Blender add-on that converts selected mesh objects into DICOM outputs for synthetic CT/MR image series, camera-based digitally reconstructed radiographs (DRRs), RT Dose grids, and RT Structure Sets. It voxelizes the active mesh selection directly into modality-appropriate intensities, supports single-phase or 4D acquisitions, and layers in synthetic artifacts tailored to the chosen modality for training or visualization workflows.

> [!WARNING]
> DICOMator produces synthetic research, education, and visualization data. It is not a medical device, is not validated for clinical diagnosis or treatment, and must not be imported into a clinical workflow as patient-acquired data. Every generated SOP instance is marked `SyntheticData = YES`.

## Examples

Sample outputs produced by the add-on:

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b1c62567-4189-4a66-812f-005b57629184" alt="Synthetic CT series generated from a skull mesh" width="320" />
      <br /><sub>Synthetic CT series (skull)</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/77e204bd-2a70-46bb-af8f-c3327ef7eb8f" alt="Synthetic CT series generated from a lung phantom" width="320" />
      <br /><sub>Synthetic CT series (lung)</sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/2951918f-773b-4505-88cc-4086dfd64b2c" alt="Camera-based digitally reconstructed radiograph of Suzanne" width="320" />
      <br /><sub>Camera-based DRR (Suzanne)</sub>
    </td>
  </tr>
</table>

## Features

- **Per-object DICOM type and intensities**
  - Each selected mesh is tagged as **Image**, **RT Dose**, or **RT Structure** via the Objects panel
  - Image objects: set HU/intensity value or pick a tissue preset; an explicit overlap priority controls which mesh wins
  - RT Dose objects: assign an absorbed dose value (Gy) per mesh; voxels within the mesh receive that dose
  - RT Structure objects: assign an ROI type (GTV, CTV, PTV, OAR, External, Control, Avoidance, Organ, Treated Volume, Irradiated Volume); contours are extracted at each image slice plane
- **Tissue intensity presets**
  - The **Material Presets** selector at the top of the Objects panel chooses CT, T1 MR, or T2 MR; it drives both the preset intensities and whether the image series is written as CT or MR
  - Eighteen presets cover air, lung, fat, soft tissue, abdominal organs, cartilage, blood, brain matter, CSF, bone, and metal implants (see the table below)
  - Selecting a preset populates that object's intensity automatically, while `Custom` leaves the manually entered value untouched
  - Changing the modality re-applies every preset intensity in the scene
- **Single-phase or 4D export**
  - Export the current frame or a range of frames (timeline or custom range)
  - One `SeriesInstanceUID` per phase; phases are written as separate series with temporal DICOM tags (`NumberOfTemporalPositions`, `TemporalPositionIndex`, `TemporalPositionIdentifier`)
  - Timeline advances during 4D export and a fixed padded bounding box keeps grids aligned between phases
- **Camera-based DRR generation**
  - Enable DRR output alongside, or instead of, the image series output
  - The DRR uses a monoenergetic Beer-Lambert approximation with a configurable effective water attenuation coefficient in `m^-1`
  - Detector size follows the Blender render resolution (including the resolution percentage) with an optional DRR resolution scale
  - Orthographic cameras emit patient-space detector geometry; perspective images omit spatial tags whose detector scale would be ambiguous
- **Voxelization control**
  - Independent lateral (XY) and axial (Z) voxel size in millimetres
  - Optional evaluation of modifiers/shape keys/armatures during voxelization
  - BVH-based +Z column fill for solid voxelization with consistent grid dimensions
  - Only the columns covering each mesh's XY footprint are ray-cast, so small meshes inside a large grid stay cheap
- **Synthetic artifact suite** (image series only)
  - The **Artifacts** sub-panel sits under **Export** and is shown only while **Image** output is enabled; artifacts never reach the DRR, RT Dose, or RT Structure outputs
  - CT applies, in order: partial volume (scanner point-spread), projection-domain metal streaks (Radon forward/back-projection with photon starvation and beam hardening), detector-channel rings, motion blur, Gaussian noise, and quantum (Poisson) noise
  - MRI applies, in order: motion blur, geometric distortion (gradient non-linearity and B0 off-resonance), Gibbs/truncation ringing, coil-shaped bias-field shading, and Rician noise, after which the magnitude volume is clamped to non-negative values
  - The single **Gaussian** noise toggle switches to Rician noise for the MR modalities, reusing the entered value as the per-channel standard deviation
  - A stored artifact seed makes repeated exports reproducible and gives every 4D phase a deterministic sub-seed
- **Patient and orientation metadata**
  - Patient Name, MRN (Patient ID), Birth Date (YYYYMMDD), Sex, and Patient Position (HFS/FFS/HFP/FFP/HFDR/HFDL/FFDR/FFDL)
  - Study ID and Accession Number are user-editable and written to every exported object
  - UTF-8 metadata is declared explicitly, value delimiters/control characters are sanitized, and DICOM byte limits are enforced
  - Customizable Series Description per export or phase
  - All objects exported together share a single study timestamp, so Study/Series/Content dates and times agree across image slices, DRR, dose, plan, and structure files
- **Export path handling and progress feedback**
  - Accepts Blender-relative paths starting with `//` (resolved relative to the `.blend` file or current working directory)
  - Defaults to `//DICOM_Export` so a new install starts with a portable export location instead of an OS-specific absolute path
  - Exports run as a modal background job: the Blender UI stays responsive, a progress bar tracks voxelization and slice writing, and **ESC cancels** the export at any time
  - Exports are staged in a temporary sibling directory and committed only after every requested object succeeds; cancel or failure removes the partial staging directory
  - A new or empty export directory is required, preventing stale files and mixed Study/Series UIDs
  - Meshes with no faces or lying outside the grid are skipped with a warning, and evaluated meshes with non-manifold geometry or ambiguous odd ray intersections raise warnings of their own
- **RT Dose export**
  - Mesh objects tagged as RT Dose are voxelized and written as a single multi-frame DICOM RT Dose file (`RTDoseStorage`)
  - Dose values (Gy) are encoded as uint32 scaled by a `DoseGridScaling` factor computed from the peak dose in the grid
  - Overlapping dose meshes either **sum** (default) or **overwrite** using explicit overlap priority
  - Configurable `DoseType` (Physical / Effective); `DoseSummationType` is restricted to Plan until real fraction and treatment-beam geometry are implemented
  - A minimal unapproved companion RT Plan file (`RTPlanStorage`) is written and referenced via `ReferencedRTPlanSequence`
  - RT Dose, image, DRR, and RT Structure exports share the same Study Instance UID and Frame of Reference UID when enabled together
- **RT Structure Set export**
  - Mesh objects tagged as RT Structure are sliced at each image Z-plane using bmesh bisection
  - Closed planar contours are extracted from cut edges and written as a DICOM RT Structure Set (`RTStructureSetStorage`)
  - ROI display colour is read from the object's first material diffuse colour, or a clinical-style palette is used as a fallback
  - Supports ROI types GTV, CTV, PTV, OAR, External, Control, Avoidance, Organ, Treated Volume, and Irradiated Volume
- **Selection insights**
  - Live estimates of grid dimensions, voxel counts, approximate peak memory usage, and DRR detector size before export

<details>
<summary><strong>Tissue preset intensities</strong></summary>

| Preset | CT (HU) | T1 MR | T2 MR |
| --- | ---: | ---: | ---: |
| Air | -1000 | 0 | 0 |
| Cortical Bone / Calcification | 1100 | 10 | 8 |
| Trabecular Bone / Fatty Marrow | 200 | 190 | 170 |
| Fat (subcutaneous, orbital) | -75 | 210 | 170 |
| Muscle | 50 | 90 | 80 |
| Liver | 50 | 100 | 90 |
| Spleen | 50 | 110 | 120 |
| Kidney Cortex | 40 | 110 | 100 |
| Kidney Medulla | 40 | 90 | 150 |
| Cartilage | 200 | 100 | 130 |
| Blood (acute, deoxyHb) | 60 | 130 | 30 |
| White Matter | 25 | 150 | 50 |
| Gray Matter | 40 | 110 | 110 |
| CSF / Water / Edema | 0 | 30 | 230 |
| Lung Parenchyma | -700 | 20 | 80 |
| Soft Tissue | 40 | 100 | 90 |
| Aluminium | 300 | 0 | 0 |
| Titanium (implant) | 3000 | 0 | 0 |

</details>

## Requirements

- Blender 5.1+ (Python 3.13 runtime); `blender_version_min` in the extension manifest is `5.1.0`
- NumPy (bundled with Blender, used for grid operations)
- pydicom 3.0.2 (bundled as an unmodified PyPI wheel and installed automatically by Blender)

## Blender compatibility

- Targets the Blender 5.1 Python 3.13 runtime (VFX Platform 2026); CI additionally smoke-tests registration on the current stable series.
- The add-on has **no OpenVDB dependency**. Voxelization is performed with Blender mesh evaluation, `mathutils.bvhtree.BVHTree`, and NumPy arrays.
- NumPy 2.x compatibility is maintained by using `np.asarray(...)` rather than `np.array(..., copy=False)` throughout.
- Extension packaging advertises only the vendored `pydicom 3.0.2` (`py3-none-any`) wheel, which requires Python 3.10+ and is compatible with NumPy 2.x and Python 3.13.
- `blender_manifest.toml` is the authoritative release metadata; the add-on carries no legacy `bl_info` block.

## Installation

Once DICOMator is listed on the official Blender Extensions platform:

1. Open **Edit → Preferences → Get Extensions**.
2. Search for **DICOMator**.
3. Select **Install**.

To install a development or pre-release build from this repository:

1. Build the extension from the repository root:

   ```bash
   blender --command extension validate
   blender --command extension build --output-filepath dicomator.zip
   ```

2. In Blender, open **Edit → Preferences → Extensions**.
3. Open the Extensions menu and choose **Install from Disk...**.
4. Select `dicomator.zip`.

The built package is self-contained. Blender installs its bundled pydicom wheel; users do not need to modify Blender's Python environment.

## Usage

1. Select one or more mesh objects in the 3D Viewport, in Object Mode.
2. In **Sidebar → DICOMator**, configure the panels:
   - The main **DICOMator** panel shows the selected object mix, the output checkboxes, and the export button. When something blocks the export (missing pydicom, a non-empty export folder, no scene camera for a DRR, a wrong unit scale, an oversized grid) the reason is shown in place of, or above, the button.
   - **Objects** – Pick the **Material Presets** modality (CT, T1 MR, or T2 MR), then for each selected mesh choose its **DICOM Type**:
     - *Image*: assign HU/intensity values or pick modality-aware tissue presets. Set **Overlap Priority** when meshes overlap; the highest value wins, with names used only as a deterministic tie-breaker.
     - *RT Dose*: assign an absorbed dose in Gy and an overlap priority. Voxels within the mesh receive that dose value when the RT Dose grid is built.
     - *RT Structure*: assign an ROI type (GTV, CTV, PTV, OAR, External, Control, Avoidance, Organ, Treated Volume, Irradiated Volume). The object's material diffuse colour is used as the ROI display colour in the structure set; a clinical-style palette is used if no material is assigned.
   - Enable the desired outputs in the main panel:
     - **Image** – writes CT or MR slices from Image meshes
     - **DRR** – writes a camera-based projection from Image meshes
     - **Dose** – writes RT Dose from RT Dose meshes
     - **Structures** – writes RT Structure Set from RT Structure meshes
   - **Export**
     - Configure **Lateral (mm)** and **Axial (mm)** voxel spacing
     - Toggle **Apply Modifiers** to evaluate modifiers, armatures, shape keys, and lattices during voxelization
     - Toggle **Allow Oversized Grids** to bypass the grid guardrails at your own risk
     - When DRR is enabled, set **DRR Resolution Scale** and the effective **Water Attenuation (1/m)** used by the monoenergetic approximation; the resolved camera and detector size are reported here
     - When any RT Dose mesh is selected, dose settings appear for **Dose Type**, plan-level **Summation Type**, and **Dose Overlap**
     - Choose a new or empty **Export Directory** (supports `//` relative paths and defaults to `//DICOM_Export`; the resolved absolute path is displayed)
     - Toggle **Export 4D/Time Series** to export multiple frames
       - Use the timeline range or set a custom `Start`/`End`/`Frame Step`
     - **Artifacts** – Nested under Export, collapsed by default, and only available while Image output is enabled:
       - Set **Artifact Seed** to reproduce random artifact fields and noise
       - *CT*: Gaussian noise, partial volume, projection-domain metal streaks, rings, quantum noise, and motion
       - *MRI (T1/T2)*: Gaussian (applied as Rician) noise, coil bias-field shading, geometric distortion (gradient non-linearity + B0 off-resonance), Gibbs ringing, and motion
   - **Series** – Set the series description, patient name, MRN, birth date, sex, patient position, study ID, and accession number in one place.
   - **Estimate** – Inspect selection size, estimated grid resolution, voxel count, conservative peak memory, and DRR detector dimensions.
3. Click **Export DICOM**.
   - For single-phase exports the mesh selection is voxelized once and written directly in HU/intensity values.
   - For 4D exports the timeline advances through the configured frame range, re-voxelizing each phase inside a fixed padded bounding box so every phase shares identical grid dimensions. Each phase receives its own Series Instance UID and the series description is suffixed with the phase number and percent completion.
   - When DRR is enabled, the voxelized HU volume is projected from the active camera into a single DICOM secondary-capture image per phase. CT presets are recommended because the DRR attenuation model assumes HU-like values.
   - When Dose or Structures are enabled, matching per-object meshes are exported alongside the image/DRR outputs in the same output directory, all sharing the same Study Instance UID and Frame of Reference UID.

Notes:
- DICOMator interprets one Blender unit as one metre. Set **Scene Properties → Units → Unit Scale** to `1.0`; export is blocked for other scales.
- Blender world axes map directly to patient coordinates: `+X` left, `+Y` posterior, and `+Z` superior.
- During 4D export the timeline visibly advances; keep animation drivers and dependencies evaluated.
- Only one export may run at a time; the export button is unavailable while a job is in progress, in Edit Mode, or without an active mesh object.
- When using relative (`//`) paths, save your `.blend` file so the path resolves predictably.

## Output details

Every enabled output is written as its own DICOM series inside a single study:

| Output | Single phase | 4D phase |
| --- | --- | --- |
| Image series | `CT_Slice_0001.dcm` / `MR_Slice_0001.dcm` … | `Phase_001_CT_Slice_0001.dcm` … |
| DRR | `DRR_Image_0001.dcm` | `Phase_001_DRR.dcm` |
| RT Dose | `RTDose.dcm` | `Phase_001_RTDose.dcm` |
| RT Plan (companion) | `RTPlan.dcm` | `Phase_001_RTPlan.dcm` |
| RT Structure Set | `RTStruct.dcm` | `Phase_001_RTStruct.dcm` |

- **Image series output**
  - Modality: CT (`CTImageStorage`) or MR (`MRImageStorage`) depending on the selected imaging modality
  - Data type: int16 signed (direct HU/intensity values)
  - Background voxels are -1000 HU (air) for CT and 0 (signal void) for MR
  - `ImageType` identifies the images as derived (`DERIVED\PRIMARY\AXIAL` for CT, `DERIVED\PRIMARY\OTHER` for MR); `SyntheticData = YES` and `SpecificCharacterSet = ISO_IR 192` are written to every slice
  - CT slices carry the rescale attributes (`RescaleIntercept = 0`, `RescaleSlope = 1`, `RescaleType = HU`) and a representative `KVP` of 120; the MR IOD omits them
  - MR series include MR Image module attributes matched to the selected weighting preset (T1: spin-echo TR 500/TE 15; T2: fast spin-echo TR 4000/TE 100, echo train 16, at 1.5 T) so the metadata is consistent with the intensities
  - Window: CT exports default to Center 40 / Width 400; MR exports use Center 128 / Width 256
  - Geometry:
    - `PixelSpacing = [voxel_size_mm_y, voxel_size_mm_x]`
    - `SliceThickness = SpacingBetweenSlices = voxel_size_mm_z`
    - `ImageOrientationPatient = [1,0,0, 0,1,0]` (axial, aligned to world axes)
    - `ImagePositionPatient` is the centre of the first transmitted voxel, i.e. the padded world-space bounding box origin plus half a voxel
    - Decimal String values are re-formatted to fit the 16-byte DICOM limit without losing millimetre precision
- **RT Dose output**
  - Storage class: `RTDoseStorage` (SOP class 1.2.840.10008.5.1.4.1.1.481.2)
  - Data type: uint32 multi-frame image scaled by `DoseGridScaling` (Gy/count); maximum dose maps to the full uint32 range
  - Grid dimensions and spatial coordinates match the image grid exactly, ensuring voxel-to-voxel correspondence
  - `DoseType`, plan-level `DoseSummationType`, and `DoseUnits = GY` are set from the RT Dose settings; `FrameIncrementPointer` references `GridFrameOffsetVector`
  - A minimal companion RT Plan (`RTPlan.dcm` / `Phase_###_RTPlan.dcm`) is written as `UNAPPROVED` and referenced via `ReferencedRTPlanSequence`
  - Shares `StudyInstanceUID` and `FrameOfReferenceUID` with co-exported image, DRR, and structure outputs
- **RT Structure Set output**
  - Storage class: `RTStructureSetStorage` (SOP class 1.2.840.10008.5.1.4.1.1.481.3)
  - Contours generated by bisecting each structure mesh at every image Z-plane using bmesh operations
  - Only closed planar loops with three or more points are written as `CLOSED_PLANAR` contours; open or degenerate edge chains are discarded and reported as warnings
  - Each ROI's colour is taken from the object's material diffuse colour (0–255 RGB); a clinical-style palette provides fallback colours
  - References the co-exported image series (series, SOP class, and per-slice SOP instances) when Image output is enabled
- **DRR output**
  - Storage class: Secondary Capture (`SecondaryCaptureImageStorage`) with `ImageType = DERIVED\PRIMARY\DRR`, `Modality = OT`, and `ConversionType = SYN`
  - Data type: uint16 monochrome projection image
  - HU is converted to linear attenuation with `mu = mu_water × (1 + HU/1000)` before ray integration
  - Single-phase exports use percentile auto-windowing; 4D exports use a fixed physical mapping so intensities stay comparable across phases
  - Geometry:
    - For orthographic cameras, `PixelSpacing` follows the world-space detector dimensions divided by detector pixels
    - `ImageOrientationPatient` follows the detector row/column axes
    - `ImagePositionPatient` is the centre of the detector's first pixel, including the required half-pixel offsets
    - Perspective-camera output omits these patient-space geometry tags because Blender's view-frame plane does not define a physical detector distance and scale
- Temporal DICOM tags (4D only):
  - `NumberOfTemporalPositions` (total phases)
  - `TemporalPositionIndex` (1-based phase order)
  - `TemporalPositionIdentifier` (1-based phase order)

## Performance and limits

- Guardrails abort exports of extremely large grids with an error:
  - Per-dimension limit: 2,000 voxels
  - Total voxels limit: 100,000,000
  - Conservative estimated peak array memory limit: 2 GiB, including RT Dose, DRR, artifact, and FFT temporaries
  - Enable **Allow Oversized Grids** in the Export panel to bypass the limits at your own risk (oversized exports may be very slow or run out of memory)
- Tips:
  - Increase voxel spacing (mm) to reduce memory/time requirements
  - Reduce the number of selected meshes or animation frames
  - Disable modifier evaluation if you only need undeformed geometry
  - Narrow the frame range or increase frame step for 4D exports
  - Disable artifact generators when testing baseline exports

## Known limitations

- Voxelization is axis-aligned and uses +Z column fills; only mesh geometry is sampled (materials/textures are ignored, apart from the RT Structure display colour).
- Deterministic sub-voxel retry rays reduce edge/vertex grazing failures, but complex coincident surfaces may still require mesh repair.
- DRR generation projects the voxelized volume rather than the original triangle mesh, so image sharpness depends on the chosen voxel spacing.
- Modifier/armature evaluation is optional but increases memory/time usage; complex rigs may still require baking.
- Output orientation is fixed to axial slices aligned with Blender world axes, with one Blender unit required to equal one metre.
- DRR export requires an active scene camera and uses image-type meshes without the CT/MR artifact stack.
- RT Structure Set contour extraction is performed per-slice using planar bisection; very thin or highly curved structures may produce incomplete contours at coarse voxel spacings.
- RT Dose is plan-level synthetic data only. Fraction and beam dose are unavailable until DICOMator models the required fraction scheme and treatment geometry.
- Each mesh contributes to exactly one DICOM object type; use duplicate meshes if the same geometry should be exported as multiple object types.
- Only the modality-specific artifacts listed above are available; additional acquisition effects are not modeled in this release.

## Troubleshooting

- **“pydicom library not available”**
  - Reinstall the DICOMator extension package. Its pydicom dependency is bundled and should be installed automatically by Blender.
- **“Voxel grid too large”**
  - Increase voxel spacing (mm), reduce padding/selection size, or limit the frame range. To proceed anyway, enable **Allow Oversized Grids** in the Export panel.
- **“Set an active scene camera before exporting a DRR”**
  - Assign a camera to the scene (`Scene Properties → Camera`) or make a camera active in the 3D View before DRR export.
- **“Choose a new or empty export folder”**
  - DICOMator never mixes a new study with existing files. Choose an empty directory or move the previous export elsewhere.
- **“Scene Unit Scale must be 1.0”**
  - One Blender unit is interpreted as one metre. Reset **Scene Properties → Units → Unit Scale** to `1.0`.
- **Export button is missing or greyed out**
  - An export is already running, the active object is not a mesh, or Blender is not in Object Mode.
- **Artifacts look too strong/weak**
  - Adjust intensity/severity controls or disable individual artifact toggles to isolate effects.
- **RT Structure contours missing or incomplete**
  - Ensure structure meshes are closed (manifold) and intersect the image grid Z-planes. Increase axial resolution (decrease voxel spacing) to capture thin structures.
- **RT Dose grid is all zeros**
  - Verify that the mesh objects intended as dose volumes have their DICOM Type set to *RT Dose* and that a non-zero Dose (Gy) value is assigned in the Objects panel.

## Repository layout

| Path | Purpose |
| --- | --- |
| `__init__.py` | Blender registration entry point, per-object properties, and exported API |
| `properties.py` | Scene-level `PropertyGroup` backing the add-on UI |
| `panels.py` | 3D Viewport sidebar panels (DICOMator, Objects, Export, Artifacts, Series, Estimate) |
| `operators.py` | Modal export operator and the staged export pipeline |
| `voxelization.py` | BVH ray-cast voxelization and selection bounds helpers |
| `artifacts.py` | Synthetic CT/MR artifact generators |
| `dicom_export.py` | CT/MR slice and DRR secondary-capture writers |
| `drr.py` | Camera-based DRR ray casting and detector geometry |
| `rtdose_export.py` | RT Dose (and companion RT Plan) writer |
| `rtstruct_export.py` | Mesh-to-contour slicing and RT Structure Set writer |
| `constants.py` | HU constants, tissue presets, DICOM helpers, and export guardrails |
| `utils.py` | Property access and export path resolution helpers |
| `wheels/`, `download_wheels.py` | Vendored pydicom wheel declared in the manifest, and its refresh script |
| `tests/` | Headless pytest suite plus a Blender registration smoke test |

## Development and testing

- A headless test suite lives in `tests/` and runs without Blender (stub `bpy`/`bmesh`/`mathutils` modules are installed by `tests/conftest.py`):

  ```bash
  pip install -r requirements-test.txt
  pytest -q
  ruff check .
  python -m compileall -q .
  ```

- Extension packaging can be validated and built with Blender itself:

  ```bash
  blender --command extension validate
  blender --command extension build --output-filepath dicomator.zip
  ```

- Continuous integration runs byte-compilation, lint, DICOM/numeric tests, and real Blender registration smoke tests (`tests/blender_smoke.py`) against the minimum supported and current stable Blender series.
- It also runs Blender's official extension validator and builds the distributable archive. Maintainers can refresh the unmodified PyPI wheel with `download_wheels.py` when updating the pinned pydicom version.

## License

This project is released under the GNU General Public License v3.0 or later, as required for add-ons distributed on the official Blender Extensions platform. See [LICENSE](LICENSE) for details. The bundled pydicom wheel retains its upstream MIT license.
