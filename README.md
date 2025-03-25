## DICOMator: Blender Addon for Synthetic CT Data Generation

**Overview**

The DICOMator is a Blender addon designed to convert selected mesh objects into realistic, synthetic CT (Computed Tomography) datasets. It includes various CT artifact simulations, making it ideal for creating training, testing, or simulation data for medical imaging applications.

**Features**

* **Adaptive Voxelization:** Convert meshes into 3D volumes with intelligent boundary detection and adaptive sampling.
* **Real-time Progress Tracking:** Watch the voxelization process with status updates and a progress indicator.
* **DICOM Export:** Export voxelized data as a series of DICOM CT images compatible with medical imaging software.
* **4D CT Series Generation:** Create 4D CT datasets simulating motion (e.g., breathing) across multiple frames.
* **Artifact Simulation:**
    * **Gaussian Noise:** Add realistic noise to CT images.
    * **Metal Artifacts:** Simulate radial streaks around high-density objects.
    * **Partial Volume Effects:** Apply smoothing to mimic artifacts caused by overlapping structures.
    * **Ring Artifacts:** Introduce ring artifacts commonly seen in CT imaging.
* **Custom Densities and Priorities:** Assign specific Hounsfield Unit (HU) values and overlapping priorities to mesh objects.

**Installation**

1. **Download the Extension:** Download the DICOMator extension package.
2. **Install the Extension in Blender:**
    * Go to Edit > Preferences > Add-ons.
    * Click "Install..." and select the downloaded zip file.
    * Enable the DICOMator checkbox.
3. **Dependencies:** All required dependencies are included in the package as wheels and will be installed automatically.
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
    * **Sampling Density:** Controls the maximum number of samples per dimension at object boundaries (higher = more accurate but slower).
    * **4D CT Export:** Enable to export over multiple frames.
        * Set Start and End Frames.
    * **Artifact Simulation:** Configure noise, metal artifacts, partial volume effects, and ring artifacts.

![metal_streaks](https://github.com/user-attachments/assets/714b3bbc-3e8e-442d-bd39-ab766e1b1cfa)

**Running the Voxelization**

1. Click "Voxelize Selected Objects" after configuration.
2. A progress indicator will appear in Blender's status bar showing the current operation and completion percentage.
3. The addon processes meshes using an efficient adaptive sampling approach:
   * First pass: Detects boundary regions using a fast, low-resolution scan
   * Second pass: Applies high-resolution sampling only where needed (at object boundaries)
4. When complete, DICOM series is exported to the specified output directory (default: "DICOM_Output" folder).

**Output**

* DICOM Series: Compatible with medical imaging software.
* File Naming: Files are named based on phase and slice number (e.g., CT_Phase_1_Slice_0001.dcm).

![skull_multi](https://github.com/user-attachments/assets/b1c62567-4189-4a66-812f-005b57629184)

![skull_dose_lat](https://github.com/user-attachments/assets/eca22ede-4a6f-47ca-a82c-e53dccb0649d)

![Lung_geometry](https://github.com/user-attachments/assets/8eb7a3ce-fbaf-4d7d-b70d-33e7b808e0fd)

![Lung](https://github.com/user-attachments/assets/77e204bd-2a70-46bb-af8f-c3327ef7eb8f)

**Notes and Tips**

* **Adaptive Sampling:** The addon intelligently detects object boundaries and applies more samples only where needed, significantly improving performance.
* **Performance Optimization:** For faster results:
  * Increase the voxel size (lower resolution)
  * Reduce the sampling density (fewer samples at boundaries)
  * Use simple mesh objects with clean geometry
* **Memory Usage:** Large scenes with small voxel sizes may require significant RAM. Consider using larger voxel sizes for initial tests.
* **Progress Tracking:** Watch the status bar for real-time progress information. You can press ESC to cancel a long-running operation.
* **Output Directory:** Customizable through the panel. Ensure you have write permissions to the selected directory.

**Troubleshooting**

* **Addon Not Appearing:** Ensure installation and enabled status in preferences. Check console for installation errors.
* **Slow Performance:** Increase voxel size or reduce sampling density. Consider simplifying complex meshes.
* **No Visible Artifacts:** Make sure artifact settings have high enough values to be visible. For metal artifacts, ensure your objects have density values above the metal threshold.
* **Noise Not Visible:** Try increasing the noise standard deviation value.
* **Export Issues:** Verify write permissions for the output directory. Check console for file saving errors.
* **Processing Cancellation:** Press ESC to cancel a long-running voxelization process.

**Contributing**

* Feedback and improvements are welcome at the GitHub repository: https://github.com/drmichaeldouglass/DICOMator
