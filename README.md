# DICOMator

**Blender add-on for synthetic CT dataset generation**

## Overview

DICOMator converts selected Blender mesh objects into realistic CT volumes. The add-on simulates common CT artifacts and exports the result as DICOM files, making it useful for generating data for training, testing, or simulation in medical-imaging research.

## Features

- Adaptive voxelization with boundary-aware sampling
- Real-time progress indicator
- Export to DICOM image series
- 4D CT series for motion studies
- Artifact simulation:
  - Gaussian noise
  - Metal-induced streaks
  - Partial-volume blur
  - Ring artifacts
- Custom Hounsfield units and overlap priority per object

## Repository Layout

- `__init__.py` – main Blender add-on module
- `blender_manifest.toml` – manifest so Blender recognises the add-on
- `wheels/` – bundled dependency wheels (SciPy, scikit-image, pydicom, …)
- `download_wheels.py` – helper script to refresh wheels for other platforms

## Installation

1. Download the DICOMator `.zip` package.
2. In Blender open **Edit → Preferences → Add-ons**.
3. Click **Install…**, choose the downloaded zip, then enable **DICOMator**.
4. Bundled wheels from `wheels/` are installed automatically when the add-on is enabled.  
   Run `download_wheels.py` to update or fetch wheels for other platforms.
5. Optionally click **Save Preferences** to keep the add-on enabled.

## Usage

### Preparing a scene

1. Select the meshes you want to convert.
2. (Optional) In the DICOMator panel set Density (HU) and Priority for overlaps.  
   Operators **Set Default Density** and **Set Default Priority** apply values to multiple objects.

![GUI](https://github.com/user-attachments/assets/55b11a36-33bb-4c42-84bc-facb3e311efd)

### Voxelization settings

In the panel you can configure:

- **Voxel Size (mm)** – smaller values increase resolution and compute time
- **Sampling Density** – maximum samples per dimension at object boundaries
- **4D CT Export** – enable and set start/end frames to export across frames
- **Artifact Simulation** – control noise, metal streaks, partial-volume blur and rings

![metal_streaks](https://github.com/user-attachments/assets/714b3bbc-3e8e-442d-bd39-ab766e1b1cfa)

### Running

1. Click **Voxelize Selected Objects**.
2. Progress appears in Blender's status bar.
3. DICOM files are written to the selected output directory (default `DICOM_Output`).

## Output

- DICOM series compatible with standard medical-imaging tools
- Files named by phase and slice, e.g. `CT_Phase_1_Slice_0001.dcm`

![skull_multi](https://github.com/user-attachments/assets/b1c62567-4189-4a66-812f-005b57629184)
![skull_dose_lat](https://github.com/user-attachments/assets/eca22ede-4a6f-47ca-a82c-e53dccb0649d)
![Lung_geometry](https://github.com/user-attachments/assets/8eb7a3ce-fbaf-4d7d-b70d-33e7b808e0fd)
![Lung](https://github.com/user-attachments/assets/77e204bd-2a70-46bb-af8f-c3327ef7eb8f)

## Tips

- Adaptive sampling focuses computation at object boundaries for efficiency.
- For faster runs:
  - Increase voxel size
  - Decrease sampling density
  - Use simplified meshes
- Large scenes with small voxels can consume significant RAM.
- Press `ESC` to cancel a long computation.
- Ensure the output directory is writable.

## Troubleshooting

- **Add-on missing** – verify installation and that the add-on is enabled.
- **Slow performance** – increase voxel size or reduce sampling density; simplify meshes.
- **Artifacts not visible** – raise settings, and for metal streaks ensure objects exceed the metal-density threshold.
- **Noise not visible** – raise the noise standard deviation.
- **Export failures** – check write permissions for the output directory.

## Contributing

Feedback and improvements are welcome at [github.com/drmichaeldouglass/DICOMator](https://github.com/drmichaeldouglass/DICOMator).

