# DICOMator
A Blender Add-On for making synthetic CT datasets from 3D meshes

Overview
DICOMator is a Blender addon that converts selected mesh objects into synthetic CT (Computed Tomography) datasets with various CT artifacts. This tool is ideal for generating realistic medical imaging data for simulation, training, or testing purposes.

Features
Voxelization of Meshes: Convert selected meshes into voxelized volumes with customizable voxel sizes.
DICOM Export: Export voxelized data as a series of DICOM CT images compatible with medical imaging software.
4D CT Series Generation: Create 4D CT datasets over a range of frames to simulate motion (e.g., breathing cycles).
Artifact Simulation:
Gaussian Noise: Add realistic noise to CT images.
Metal Artifacts: Simulate radial streaks around high-density objects.
Partial Volume Effects: Apply smoothing to mimic partial volume artifacts.
Ring Artifacts: Introduce ring artifacts commonly seen in CT imaging.
Beam Hardening: Simulate beam hardening effects.
Custom Densities and Priorities: Assign specific density values and overlapping priorities to mesh objects.
Voxelization Methods:
Fast (Ray Casting): Quick voxelization using ray casting.
Accurate (Sampling): Precise voxelization using volume sampling.
Installation
1. Download the Addon
Save the provided addon script as a .py file, e.g., voxelizer_extension.py.
2. Install Dependencies
The addon requires the following Python packages:

pydicom
scikit-image
The addon attempts to install pydicom automatically. However, you need to install scikit-image manually.

Installing scikit-image:
Open Blender and go to the Python Console or use your system's terminal.

Run the following command:

python
Copy code
import bpy
import subprocess
subprocess.check_call([bpy.app.binary_path_python, "-m", "pip", "install", "scikit-image"])
Wait for the installation to complete.

3. Install the Addon in Blender
Open Blender.
Go to Edit > Preferences > Add-ons.
Click on Install... at the top.
Navigate to and select the voxelizer_extension.py file.
Enable the addon by checking the box next to Voxelizer Extension.
4. Save User Preferences (Optional)
Click Save Preferences to keep the addon enabled for future Blender sessions.
Usage
Preparing Your Scene
Select Mesh Objects:

In the 3D Viewport, select the mesh objects you wish to voxelize.
Only selected mesh objects will be processed.
Assign Densities and Priorities (Optional):

With a mesh object selected, open the Voxelizer panel in the sidebar (press N if the sidebar is hidden).

Under Object Properties, set:

Density: The Hounsfield Unit (HU) value to assign to the object.
Priority: Determines which object's density takes precedence in overlapping regions.
To set default values for multiple objects:

Select the desired objects.
Use the Set Default Density and Set Default Priority operators in the panel.
Configuring Voxelization Settings
Open the Voxelizer panel and adjust the following settings:

Voxelization Settings
Voxel Size (mm): Size of each voxel in millimeters. Smaller sizes yield higher resolution but increase computation time.
Enable 4D CT Export: Toggle to export over a range of frames.
Start Frame: The first frame to export.
End Frame: The last frame to export.
Artifact Simulation
Add Noise:
Noise Std Dev: Standard deviation of the Gaussian noise.
Simulate Metal Artifacts:
Metal Threshold: Density threshold to identify metal voxels.
Streak Intensity: Intensity of the simulated streaks.
Number of Streaks: How many streaks to generate around metal objects.
Simulate Partial Volume Effects:
PVE Sigma: Sigma value for Gaussian smoothing.
Simulate Ring Artifacts:
Ring Intensity: Intensity of ring artifacts.
Ring Frequency: Frequency of the rings.
Simulate Beam Hardening (currently commented out in the code):
Beam Hardening Metal Threshold: Density threshold for beam hardening simulation.
Voxelization Method
Voxelization Method:
Fast (Ray Casting): Quicker but less accurate.
Number of Rays per Voxel: Increase for better accuracy.
Accurate (Sampling): More precise but slower.
Samples per Voxel: Increase for better accuracy.
Running the Voxelization
Once all settings are configured, click Voxelize Selected Objects at the bottom of the Voxelizer panel.
The addon will process the meshes and export the DICOM series to a folder named DICOM_Output in the same directory as your Blender file.
Output
DICOM Series: The output is a series of DICOM files compatible with medical imaging software.
File Naming: Files are named according to the phase and slice number, e.g., CT_Phase_1_Slice_0001.dcm.
Notes and Tips
Performance:

High-resolution settings and accurate voxelization methods significantly increase processing time and memory usage.
For testing purposes, start with larger voxel sizes and fewer samples or rays.
Dependencies:

Ensure pydicom and scikit-image are properly installed.
If you encounter import errors, reinstall the packages using Blender's Python interpreter.
Customization:

The addon uses default DICOM metadata. Modify the code if you need to customize patient or study information.
Beam hardening simulation is currently commented out. You can enable it by uncommenting the relevant sections in the code.
Error Handling:

The addon reports errors in Blender's console and info bar. Check these if the addon isn't functioning as expected.
If no mesh objects are selected, the addon will cancel the operation and report an error.
Troubleshooting
Addon Not Appearing:

Ensure the addon is installed and enabled in Blender's preferences.
Check the console for any error messages during installation.
Missing Dependencies:

Reinstall pydicom and scikit-image using Blender's Python interpreter.
Verify that you are using the correct Python environment associated with Blender.
Export Issues:

Make sure the DICOM_Output directory is writable.
Check for any error messages indicating issues with file saving.
Contributing
Feedback and Improvements:

Contributions are welcome! Feel free to submit pull requests or open issues on the GitHub repository.
Bug Reports:

Provide detailed information about the issue, including steps to reproduce and any error messages.
