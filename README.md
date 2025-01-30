
## DICOMator: Blender Addon for Synthetic CT Data Generation

**Overview**

The DICOMator is a Blender addon designed to convert selected mesh objects into realistic, synthetic CT (Computed Tomography) datasets. It includes various CT artifact simulations, making it ideal for creating training, testing, or simulation data for medical imaging applications.

**Features**

* **Voxelization of Meshes:** Convert meshes into 3D volumes with customizable voxel sizes.
* **DICOM Export:** Export voxelized data as a series of DICOM CT images compatible with medical imaging software.
* **4D CT Series Generation:** Create 4D CT datasets simulating motion (e.g., breathing) across multiple frames.
* **Artifact Simulation:**
    * **Gaussian Noise:** Add realistic noise to CT images.
    * **Metal Artifacts:** Simulate radial streaks around high-density objects.
    * **Partial Volume Effects:** Apply smoothing to mimic artifacts caused by overlapping structures.
    * **Ring Artifacts:** Introduce ring artifacts commonly seen in CT imaging.
    * **Beam Hardening:** Simulate beam hardening effects (currently disabled).
* **Custom Densities and Priorities:** Assign specific Hounsfield Unit (HU) values and overlapping priorities to mesh objects.
* **Voxelization Methods:**
    * **Fast (Ray Casting):** Quick voxelization using ray casting (less accurate).
    * **Accurate (Sampling):** Precise voxelization using volume sampling (slower).

**Installation**

1. **Download the Addon:** Save the provided script as a `.py` file (e.g., `dicomator.py`).
2. **Install Dependencies:**
    * `pydicom` (automatic installation attempted)
    * `scikit-image` (manual installation required)
        * Open Blender's Python Console or your system's terminal.
        * Run: `python -m pip install scikit-image`
3. **Install the Addon in Blender:**
    * Go to Edit > Preferences > Add-ons.
    * Click "Install..." and select `dicomator.py`.
    * Enable the DICOMator checkbox.
4. **Save User Preferences (Optional):** Click "Save Preferences" to keep the addon enabled for future sessions.

**Usage**

**Preparing Your Scene**

1. **Select Mesh Objects:** Choose the meshes you want to convert in the 3D Viewport.
2. **Assign Densities and Priorities (Optional):**
    * Select a mesh object.
    * Open the DICOMator panel (press N if hidden).
    * Set Density (HU value) and Priority for overlapping regions.
    * Use "Set Default Density" and "Set Default Priority" operators for multiple objects.


![GUI](https://github.com/user-attachments/assets/55b11a36-33bb-4c42-84bc-facb3e311efd)

**Configuring Voxelization Settings**

* Open the DICOMator panel and adjust settings:
    * **Voxel Size (mm):** Size of each voxel (smaller = higher resolution, longer processing).
    * **4D CT Export:** Enable to export over multiple frames.
        * Set Start and End Frames.
    * **Artifact Simulation:** Configure noise, metal artifacts, partial volume effects, and ring artifacts.
    * **Voxelization Method:** Choose Fast (Ray Casting) or Accurate (Sampling).
        * Adjust Number of Rays/Samples per Voxel for accuracy.

![metal_streaks](https://github.com/user-attachments/assets/714b3bbc-3e8e-442d-bd39-ab766e1b1cfa)

**Running the Voxelization**

1. Click "Voxelize Selected Objects" after configuration.
2. The addon processes meshes and exports the DICOM series to a "DICOM_Output" folder in your Blender file directory.

**Output**

* DICOM Series: Compatible with medical imaging software.
* File Naming: Files are named based on phase and slice number (e.g., CT_Phase_1_Slice_0001.dcm).

![skull_multi](https://github.com/user-attachments/assets/b1c62567-4189-4a66-812f-005b57629184)

![skull_dose_lat](https://github.com/user-attachments/assets/eca22ede-4a6f-47ca-a82c-e53dccb0649d)

![Lung_geometry](https://github.com/user-attachments/assets/8eb7a3ce-fbaf-4d7d-b70d-33e7b808e0fd)

![Lung](https://github.com/user-attachments/assets/77e204bd-2a70-46bb-af8f-c3327ef7eb8f)

**Notes and Tips**

* High-resolution settings and accurate voxelization methods require more processing time and memory.
* Use larger voxel sizes and fewer samples/rays for initial testing.
* Ensure `pydicom` and `scikit-image` are installed correctly. Reinstall if needed.
* The addon uses default DICOM metadata. Modify code for patient/study information customization.
* Beam hardening simulation is currently disabled (uncomment relevant code sections for activation).
* Check Blender's console and info bar for any errors.

**Troubleshooting**

* **Addon Not Appearing:** Ensure installation and enabled status in preferences. Check console for installation errors.
* **Missing Dependencies:** Reinstall `pydicom` and `scikit-image` using Blender's Python interpreter. Verify correct Python environment.
* **Export Issues:** Verify write permissions for the "DICOM_Output" directory. Check console for file saving errors.

**Contributing**

* Feedback and improvements are welcome!
