bl_info = {
    "name": "Voxelizer Extension",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Voxelizer",
    "description": "Voxelize selected meshes and export as DICOM CT series",
    "warning": "",
    "wiki_url": "",
    "category": "Object",
}

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from bpy.types import Operator, Panel
from bpy.props import (
    FloatProperty,
    BoolProperty,
    IntProperty,
    PointerProperty,
)
from datetime import datetime
import os
import math
from skimage.draw import line

# Ensure pydicom is installed
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([bpy.app.binary_path_python, "-m", "pip", "install", "pydicom"])
    import pydicom
    from pydicom.dataset import Dataset, FileDataset


# Property Group for settings
class VoxelizerSettings(bpy.types.PropertyGroup):
    voxel_size: FloatProperty(
        name="Voxel Size (mm)",
        default=2.0,
        min=0.01,
        description="Size of each voxel"
    )

    enable_4d_export: BoolProperty(
        name="Enable 4D CT Export",
        default=False,
        description="Export a 4D CT series over a range of frames"
    )

    start_frame: IntProperty(
        name="Start Frame",
        default=1,
        min=1,
        description="Start frame for 4D export"
    )

    end_frame: IntProperty(
        name="End Frame",
        default=250,
        min=1,
        description="End frame for 4D export"
    )
    
    enable_noise: BoolProperty(
        name="Add Noise",
        default=False,
        description="Add Gaussian noise to the CT images"
    )
    noise_std_dev: FloatProperty(
        name="Noise Std Dev",
        default=20.0,
        min=0.0,
        description="Standard deviation of Gaussian noise"
    )
    # Metal artifact settings
    enable_metal_artifacts: BoolProperty(
        name="Simulate Metal Artifacts",
        default=False,
        description="Simulate metal artifacts in the CT images"
    )
    metal_threshold: FloatProperty(
        name="Metal Threshold",
        default=3000.0,
        min=0.0,
        description="Density threshold to consider voxels as metal"
    )
    streak_intensity: FloatProperty(
        name="Streak Intensity",
        default=500.0,
        description="Intensity change along the streaks"
    )
    num_angles: IntProperty(
        name="Number of Streaks",
        default=36,
        min=1,
        description="Number of radial streaks around metal objects"
    )
    # Partial volume effect settings
    enable_pve: BoolProperty(
        name="Simulate Partial Volume Effects",
        default=False,
        description="Apply smoothing to simulate partial volume effects"
    )
    pve_sigma: FloatProperty(
        name="PVE Sigma",
        default=0.5,
        min=0.0,
        description="Sigma value for Gaussian smoothing"
    )
    # Additional artifact settings (e.g., Ring Artifacts)
    enable_ring_artifacts: BoolProperty(
        name="Simulate Ring Artifacts",
        default=False,
        description="Simulate ring artifacts in the CT images"
    )
    ring_intensity: FloatProperty(
        name="Ring Intensity",
        default=10.0,
        description="Intensity of the ring artifacts"
    )
    ring_frequency: FloatProperty(
        name="Ring Frequency",
        default=5.0,
        description="Frequency of the rings"
    )

    enable_beam_hardening: BoolProperty(
        name="Simulate Beam Hardening",
        default=False,
        description="Simulate Beam Hardening artifacts in the CT images"
    )

    metal_threshold_bh: FloatProperty(
        name="Beam Hardening Metal Threshold",
        default=3000.0,
        min=0.0,
        description="Density threshold to consider voxels as metal"
    )

    voxelization_method: bpy.props.EnumProperty(
        name="Voxelization Method",
        description="Choose between fast and accurate voxelization methods",
        items=[
            ('FAST', "Fast (Ray Casting)", "Fast voxelization using ray casting"),
            ('ACCURATE', "Accurate (Sampling)", "Accurate voxelization using volume sampling")
        ],
        default='FAST'
    )
    samples_per_voxel: IntProperty(
        name="Samples per Voxel",
        default=5,
        min=1,
        description="Number of samples per voxel for accurate voxelization"
    )

    num_rays_per_voxel: IntProperty(
        name="Number of Rays per Voxel",
        default=5,
        min=1,
        description="Number of rays to cast per voxel for fast voxelization"
    )


# Operator Class
class VoxelizeOperator(Operator):
    bl_idname = "object.voxelize_operator"
    bl_label = "Voxelize Selected Objects"
    bl_options = {'REGISTER', 'UNDO'}

    # Internal attributes for UIDs
    study_instance_uid: bpy.props.StringProperty()
    frame_of_reference_uid: bpy.props.StringProperty()

    def execute(self, context):
        from pydicom.uid import generate_uid
        settings = context.scene.voxelizer_settings
        
        # Get selected meshes
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}
    
        # Voxelization parameters
        voxel_size = settings.voxel_size
        output_dir = bpy.path.abspath("//DICOM_Output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # UIDs
        study_instance_uid = generate_uid()
        frame_of_reference_uid = generate_uid()
    
        # Get the enable_4d_export setting
        enable_4d_export = settings.enable_4d_export
        start_frame = settings.start_frame
        end_frame = settings.end_frame
        scene = context.scene
        current_frame = scene.frame_current  # Store current frame to restore later
        
        if enable_4d_export:
            # Total number of frames (phases)
            if start_frame > end_frame:
                self.report({'ERROR'}, "Start Frame must be less than or equal to End Frame")
                return {'CANCELLED'}
            total_frames = end_frame - start_frame + 1
    
            for frame_index, frame in enumerate(range(start_frame, end_frame + 1), start=1):
                scene.frame_set(frame)  # Set the scene to the current frame
    
                # Generate dynamic phase data
                percentage = (frame_index / total_frames) * 100
                series_description = f"Phase {frame_index} - {percentage:.1f}% of breathing cycle"
    
                # Generate unique Series Instance UID per phase
                series_instance_uid = generate_uid()
    
                # Series number as an integer unique per phase
                series_number = frame_index
    
                # Temporal position index equal to the frame number
                temporal_position_index = frame
    
                # Compile phase information
                phase_info = {
                    'series_description': series_description,
                    'series_instance_uid': series_instance_uid,
                    'series_number': str(series_number),
                    'temporal_position_index': str(temporal_position_index),
                }
    
                # Voxelize meshes for this phase
                if settings.voxelization_method == 'FAST':
                    voxel_grid = self.voxelize_meshes(
                        selected_meshes,
                        voxel_size,
                        num_rays_per_voxel=settings.num_rays_per_voxel
                    )
                elif settings.voxelization_method == 'ACCURATE':
                    voxel_grid = self.voxelize_meshes_accurate(
                        selected_meshes,
                        voxel_size,
                        samples_per_voxel=settings.samples_per_voxel
                    )
    
                # Apply partial volume effect if enabled
                if settings.enable_pve:
                    voxel_grid = self.apply_partial_volume_effect(voxel_grid, sigma=settings.pve_sigma)
                # Simulate radial streak artifacts
                if settings.enable_metal_artifacts:
                    self.simulate_radial_streaks(
                        voxel_grid,
                        metal_threshold=settings.metal_threshold,
                        streak_intensity=settings.streak_intensity,
                        num_angles=settings.num_angles
                    )
                if settings.enable_ring_artifacts:
                    self.simulate_ring_artifacts(
                        voxel_grid,
                        intensity=settings.ring_intensity,
                        frequency=settings.ring_frequency
                    )
                if settings.enable_beam_hardening:
                    self.simulate_beam_hardening(
                        voxel_grid
                    )
        
                # Export DICOM slices for this phase
                self.export_to_dicom(
                    voxel_grid=voxel_grid,
                    voxel_size=voxel_size,
                    output_dir=output_dir,
                    study_instance_uid=study_instance_uid,
                    frame_of_reference_uid=frame_of_reference_uid,
                    noise_std_dev=settings.noise_std_dev if settings.enable_noise else 0,
                    phase_info=phase_info,
                    total_phases=total_frames,
                )
    
                self.report({'INFO'}, f"Exported phase {frame_index}/{total_frames}")
    
            # Restore the original frame
            scene.frame_set(current_frame)
        else:
            # Single frame export
            total_frames = 1  # Only one frame
            frame = scene.frame_current
            frame_index = 1
            percentage = 100.0
            series_description = f"Phase {frame_index} - {percentage:.1f}% of breathing cycle"
            series_instance_uid = generate_uid()
            series_number = frame_index
            temporal_position_index = frame
    
            phase_info = {
                'series_description': series_description,
                'series_instance_uid': series_instance_uid,
                'series_number': str(series_number),
                'temporal_position_index': str(temporal_position_index),
            }
    
            # Voxelize meshes for this frame
            if settings.voxelization_method == 'FAST':
                voxel_grid = self.voxelize_meshes(
                    selected_meshes,
                    voxel_size,
                    num_rays_per_voxel=settings.num_rays_per_voxel
                )
            elif settings.voxelization_method == 'ACCURATE':
                voxel_grid = self.voxelize_meshes_accurate(
                    selected_meshes,
                    voxel_size,
                    samples_per_voxel=settings.samples_per_voxel
                )
    
            # Apply partial volume effect if enabled
            if settings.enable_pve:
                voxel_grid = self.apply_partial_volume_effect(voxel_grid, sigma=settings.pve_sigma)
            # Simulate radial streak artifacts
            if settings.enable_metal_artifacts:
                self.simulate_radial_streaks(
                    voxel_grid,
                    metal_threshold=settings.metal_threshold,
                    streak_intensity=settings.streak_intensity,
                    num_angles=settings.num_angles
                )
            if settings.enable_ring_artifacts:
                self.simulate_ring_artifacts(
                    voxel_grid,
                    intensity=settings.ring_intensity,
                    frequency=settings.ring_frequency
                )
            if settings.enable_beam_hardening:
                self.simulate_beam_hardening(
                    voxel_grid
                )
    
            # Export DICOM slices for this frame
            self.export_to_dicom(
                voxel_grid=voxel_grid,
                voxel_size=voxel_size,
                output_dir=output_dir,
                study_instance_uid=study_instance_uid,
                frame_of_reference_uid=frame_of_reference_uid,
                noise_std_dev=settings.noise_std_dev if settings.enable_noise else 0,
                phase_info=phase_info,
                total_phases=total_frames,
            )
    
            self.report({'INFO'}, f"Exported frame {frame_index}/{total_frames}")
    
        self.report({'INFO'}, f"Voxelization complete. Output saved to {output_dir}")
        return {'FINISHED'}


    # Helper function to calculate combined bounding box
    def calculate_combined_bounding_box(self, meshes):
        """
        Calculates the combined bounding box of all selected meshes.
        
        Parameters:
        - meshes: List of Blender mesh objects.
        
        Returns:
        - bbox_min: mathutils.Vector representing the minimum corner of the bounding box.
        - bbox_max: mathutils.Vector representing the maximum corner of the bounding box.
        """
        if not meshes:
            return Vector((0, 0, 0)), Vector((0, 0, 0))
        
        # Initialize with the first mesh's bounding box
        first_mesh = meshes[0]
        bbox_min, bbox_max = first_mesh.bound_box[0], first_mesh.bound_box[6]
        bbox_min = first_mesh.matrix_world @ Vector(bbox_min)
        bbox_max = first_mesh.matrix_world @ Vector(bbox_max)
        
        # Iterate through remaining meshes to find the overall bounding box
        for mesh in meshes[1:]:
            mesh_min, mesh_max = mesh.bound_box[0], mesh.bound_box[6]
            mesh_min = mesh.matrix_world @ Vector(mesh_min)
            mesh_max = mesh.matrix_world @ Vector(mesh_max)
            bbox_min = Vector((
                min(bbox_min.x, mesh_min.x),
                min(bbox_min.y, mesh_min.y),
                min(bbox_min.z, mesh_min.z)
            ))
            bbox_max = Vector((
                max(bbox_max.x, mesh_max.x),
                max(bbox_max.y, mesh_max.y),
                max(bbox_max.z, mesh_max.z)
            ))
        
        return bbox_min, bbox_max


    def simulate_radial_streaks(self, voxel_grid, metal_threshold, streak_intensity, num_angles):
        """
        Simulates bright radial streak artifacts around high-density regions in each axial slice,
        ensuring that the streaks are confined within each slice.
    
        Parameters:
        - voxel_grid: 3D numpy array of voxel data.
        - metal_threshold: Density threshold to consider voxels as metal.
        - streak_intensity: Maximum intensity change along the streaks (positive value).
        - num_angles: Number of angles to generate streaks (e.g., 36 for every 10 degrees).
        """
        num_slices = voxel_grid.shape[2]
        for z in range(num_slices):
            # Extract the slice data for the current slice
            slice_data = voxel_grid[:, :, z].copy()  # Use a copy to avoid modifying other slices
    
            # Identify high-density (metal) voxels in the current slice
            metal_indices = np.argwhere(slice_data >= metal_threshold)
            if metal_indices.size == 0:
                continue  # No metal in this slice
    
            height, width = slice_data.shape
            for metal_voxel in metal_indices:
                x0, y0 = metal_voxel  # x0 is row index, y0 is column index
                # Generate bright streaks only
                for angle in np.linspace(0, 2 * np.pi, num=num_angles, endpoint=False):
                    # Determine the end point of the line within the slice boundaries
                    x1 = int(x0 + height * np.sin(angle))
                    y1 = int(y0 + width * np.cos(angle))
                    # Ensure end points are within the slice
                    x1 = np.clip(x1, 0, height - 1)
                    y1 = np.clip(y1, 0, width - 1)
                    # Get the coordinates of the line within the slice
                    rr, cc = line(x0, y0, x1, y1)
                    # Clip coordinates to stay within the slice
                    rr = np.clip(rr, 0, height - 1)
                    cc = np.clip(cc, 0, width - 1)
                    # Set streak intensity for this line (always positive for bright streaks)
                    intensity = streak_intensity
                    # Modify the voxel values along the line
                    distances = np.sqrt((rr - x0) ** 2 + (cc - y0) ** 2)
                    attenuation = np.exp(-distances / (0.1 * max(height, width)))
                    slice_data[rr, cc] += intensity * attenuation
            # Clip the slice data to valid HU range
            slice_data = np.clip(slice_data, -1024, 3071)
            # Replace the original slice in the voxel grid with the modified slice
            voxel_grid[:, :, z] = slice_data



    def simulate_ring_artifacts(self, voxel_grid, intensity, frequency):
        num_slices = voxel_grid.shape[2]
        center = (voxel_grid.shape[0] // 2, voxel_grid.shape[1] // 2)
        y, x = np.ogrid[:voxel_grid.shape[0], :voxel_grid.shape[1]]
        for z in range(num_slices):
            slice_data = voxel_grid[:, :, z]
            radius = np.hypot(x - center[1], y - center[0])
            rings = np.sin(2 * np.pi * frequency * radius / voxel_grid.shape[0])
            slice_data += intensity * rings
            voxel_grid[:, :, z] = slice_data


    def simulate_beam_hardening(self, voxel_grid, attenuation_coefficient=0.05):
        """
        Simulates beam hardening artifacts by applying a cupping effect to each axial slice.
    
        Parameters:
        - voxel_grid: 3D numpy array representing the voxelized volume.
        - attenuation_coefficient: Controls the strength of the beam hardening effect.
        """
        num_slices = voxel_grid.shape[2]
        for z in range(num_slices):
            # Extract the slice data
            slice_data = np.copy(voxel_grid[:, :, z])
    
            # Get the dimensions of the slice
            height, width = slice_data.shape
    
            # Calculate the center coordinates
            center_x = width / 2
            center_y = height / 2
    
            # Create a grid of x and y coordinates
            x = np.arange(width)
            y = np.arange(height)
            xx, yy = np.meshgrid(x, y)
    
            # Calculate the distance from the center for each voxel
            distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    
            # Normalize distances to range from 0 to 1
            max_distance = np.sqrt((center_x) ** 2 + (center_y) ** 2)
            normalized_distances = distances / max_distance
    
            # Calculate the cupping effect
            cupping = attenuation_coefficient * normalized_distances
    
            # Apply the cupping effect to the slice data
            slice_data *= (1 - cupping)
    
            # Update the voxel grid with the modified slice data
            voxel_grid[:, :, z] = slice_data


    def add_noise(self, voxel_grid, noise_std_dev):
        if noise_std_dev > 0:
            noise = np.random.normal(0, noise_std_dev, voxel_grid.shape)
            voxel_grid += noise

    def apply_partial_volume_effect(self, voxel_grid, sigma):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(voxel_grid, sigma=sigma)


    # Voxelization function
    def voxelize_meshes(self, meshes, voxel_size, num_rays_per_voxel=5):
        # Calculate the combined bounding box of all meshes
        bbox_min, bbox_max = self.calculate_combined_bounding_box(meshes)
        grid_size = bbox_max - bbox_min
        grid_dims = tuple(int(np.ceil(s / voxel_size)) for s in grid_size)
    
        # Initialize voxel grid with background density (e.g., -1000 for air)
        background_density = -1000.0  # Adjust as needed
        voxel_grid = np.full(grid_dims, background_density, dtype=np.float32)
    
        # For coordinate calculations in the DICOM export
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
    
        # Prepare BVH trees for all meshes
        bvh_trees = []
        densities = []
        priorities = []  # Add this line to initialize the priorities list
        for mesh in meshes:
            density = getattr(mesh, 'density', 0.0)
            priority = getattr(mesh, 'priority', 0)  # Collect the priority attribute
            # Prepare BMesh
            bm = bmesh.new()
            bm.from_mesh(mesh.data)
            bm.transform(mesh.matrix_world)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            # Create BVHTree from BMesh
            bvhtree = bvh_tree_from_bmesh(bm)
            bvh_trees.append(bvhtree)
            densities.append(density)
            priorities.append(priority)  # Append the priority to the list
            bm.free()

        # Voxelization loop
        for idx in np.ndindex(grid_dims):
            # Calculate the world coordinate of the voxel center
            point = bbox_min + Vector((
                (idx[0] + 0.5) * voxel_size,
                (idx[1] + 0.5) * voxel_size,
                (idx[2] + 0.5) * voxel_size
            ))
        
            # Find meshes containing the point
            containing_meshes = []
            for bvhtree, density, priority in zip(bvh_trees, densities, priorities):
                if self.is_point_inside_mesh(point, bvhtree, num_rays=num_rays_per_voxel):
                    containing_meshes.append({'density': density, 'priority': priority})
        
            if containing_meshes:
                # Select the mesh with the highest priority
                containing_meshes.sort(key=lambda x: x['priority'], reverse=True)
                voxel_density = containing_meshes[0]['density']
                voxel_grid[idx] = voxel_density
            else:
                # Use background density
                voxel_grid[idx] = background_density


        return voxel_grid

    def is_point_inside_mesh(self, point, bvhtree, num_rays=5):
        """
        Determines if a point is inside a mesh using multiple ray casting directions.
        
        Parameters:
        - point: mathutils.Vector representing the point to test.
        - bvhtree: BVHTree object of the mesh.
        - num_rays: int, number of rays to cast from the point.
        
        Returns:
        - True if the point is inside the mesh based on majority vote, False otherwise.
        """
        # Generate uniformly distributed ray directions
        ray_directions = generate_uniform_directions(num_rays)
        
        inside_count = 0  # Counter for rays indicating the point is inside

        for direction in ray_directions:
            current_point = point.copy()
            ray_direction = direction.normalized()
            hit_count = 0

            while True:
                location, normal, index, distance = bvhtree.ray_cast(current_point, ray_direction)
                
                if location is None:
                    break  # No more intersections

                hit_count += 1

                # Move the point slightly beyond the intersection to avoid hitting the same surface
                current_point = location + ray_direction * 1e-5

                if hit_count > 1000:
                    # Prevent infinite loops in degenerate cases
                    break

            # If the number of intersections is odd, the ray indicates the point is inside
            if hit_count % 2 == 1:
                inside_count += 1

        # Majority vote: more than half of the rays indicate inside
        return inside_count > (num_rays / 2)


    def voxelize_meshes_accurate(self, meshes, voxel_size, samples_per_voxel=5):
        # Calculate the combined bounding box of all meshes
        bbox_min, bbox_max = self.calculate_combined_bounding_box(meshes)

        # Store bbox_min and bbox_max as instance attributes
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        grid_size = bbox_max - bbox_min
        grid_dims = tuple(int(np.ceil(s / voxel_size)) for s in grid_size)
    
        # Initialize voxel grid with background density (e.g., -1000 for air)
        background_density = -1000.0  # Air in HU
        voxel_grid = np.full(grid_dims, background_density, dtype=np.float32)
    
        # Prepare BVH trees for all meshes
        bvh_trees = []
        densities = []
        for mesh in meshes:
            # Get density
            density = getattr(mesh, 'density', 0.0)
            # Prepare BMesh
            bm = bmesh.new()
            bm.from_mesh(mesh.data)
            bm.transform(mesh.matrix_world)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            # Create BVHTree from BMesh
            bvhtree = bvh_tree_from_bmesh(bm)
            bvh_trees.append(bvhtree)
            densities.append(density)
            bm.free()
    
        # Generate sample points within each voxel
        sample_offsets = np.linspace(0, voxel_size, samples_per_voxel, endpoint=False) + (voxel_size / (2 * samples_per_voxel))
        sample_points = np.array(np.meshgrid(sample_offsets, sample_offsets, sample_offsets, indexing='ij')).reshape(3, -1).T
    
        total_samples = len(sample_points)
    
        # Voxelization loop
        for idx in np.ndindex(grid_dims):
            # Voxel minimum position
            voxel_min = bbox_min + Vector((
                idx[0] * voxel_size,
                idx[1] * voxel_size,
                idx[2] * voxel_size
            ))
            # Convert voxel_min to NumPy array
            voxel_min_np = np.array(voxel_min)
            # Calculate sample points in world coordinates
            voxel_samples = voxel_min_np + sample_points
    
            occupancy = 0.0
            voxel_density = background_density
    
            # For each sample point within the voxel
            for sample in voxel_samples:
                sample_point = Vector(sample)  # Convert sample to Blender Vector
                point_inside = False
                # Check if the point is inside any of the meshes
                for bvhtree, density in zip(bvh_trees, densities):
                    if self.is_point_inside_mesh(sample_point, bvhtree):
                        point_inside = True
                        voxel_density = density  # Use the density of the first mesh that contains the point
                        break  # Stop checking other meshes
                if point_inside:
                    occupancy += 1.0
    
            # Calculate occupancy fraction
            occupancy_fraction = occupancy / total_samples
            if occupancy_fraction > 0:
                # Assign density based on occupancy fraction
                voxel_grid[idx] = occupancy_fraction * voxel_density + (1 - occupancy_fraction) * background_density
    
        return voxel_grid


    def export_to_dicom(
        self,
        voxel_grid,
        voxel_size,
        output_dir,
        study_instance_uid,
        frame_of_reference_uid,
        noise_std_dev=0,
        phase_info=None,
        total_phases=1,
        ):
        num_slices = voxel_grid.shape[2]
        for i in range(num_slices):
            slice_data = voxel_grid[:, :, i]
            result = self.save_dicom_slice(
                slice_data,
                voxel_size,
                slice_index=i,
                output_dir=output_dir,
                study_instance_uid=study_instance_uid,
                frame_of_reference_uid=frame_of_reference_uid,
                noise_std_dev=noise_std_dev,
                phase_info=phase_info,
                total_phases=total_phases,
                num_slices=num_slices,
            )
            if result != {'FINISHED'}:
                return result  # Handle cancellation if save_dicom_slice failed


    def save_dicom_slice(
        self,
        slice_data,
        voxel_size,
        slice_index,
        output_dir,
        study_instance_uid,
        frame_of_reference_uid,
        noise_std_dev=0,
        phase_info=None,
        total_phases=1,
        num_slices=1,
        ):    
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from datetime import datetime
        from pydicom.uid import generate_uid
    
        # Get current date and time
        current_datetime = datetime.now()
        date_str = current_datetime.strftime('%Y%m%d')
        time_str = current_datetime.strftime('%H%M%S.%f')  # Include fractional seconds
    
        # Create file meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    
        # Create the FileDataset instance
        ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
        # Set DICOM tags
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    
        # Patient and Study Information
        ds.PatientName = 'Dr. Smith'
        ds.PatientID = '000515054'
        ds.PatientBirthDate = ''
        ds.PatientSex = 'M'
    
        ds.StudyInstanceUID = study_instance_uid  # Remains the same for all phases
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.StudyID = '20201'
        ds.AccessionNumber = '20201'
        ds.StudyDate = date_str
        ds.StudyTime = '121656'
        ds.ReferringPhysicianName = 'Dr. Test'
    
        # Series Information
        ds.SeriesInstanceUID = phase_info.get('series_instance_uid', generate_uid())
        ds.SeriesNumber = int(phase_info.get('series_number', '1'))  # Ensure it's an int
        ds.SeriesDescription = phase_info.get('series_description', '4DCT Phase')
        ds.SeriesDate = date_str
        ds.SeriesTime = time_str
    
        # Equipment Information
        ds.Modality = 'CT'
        ds.Manufacturer = 'TOSHIBA'
        ds.InstitutionName = 'Hospital'
        ds.InstitutionAddress = 'Test Street'
        ds.StationName = 'Level 1 CT Scanner'
        ds.InstitutionalDepartmentName = 'Radiation Oncology'
        ds.ManufacturerModelName = 'Aquilion/LB'
        ds.DeviceSerialNumber = 'A7DNAHDGALS'
        ds.SoftwareVersions = 'V6.30ER012'
    
        # Image Acquisition Information
        ds.KVP = 120  # Numeric
        ds.SliceThickness = float(voxel_size)  # Numeric
        ds.DataCollectionDiameter = 700.0  # Numeric (float)
        ds.ReconstructionDiameter = 700.0  # Numeric (float)
        ds.GantryDetectorTilt = 0.0  # Numeric (float)
        ds.TableHeight = 159.0  # Numeric (float)
        ds.RotationDirection = 'CW'  # String (CS)
        ds.ExposureTime = 500  # Numeric (int)
        ds.XRayTubeCurrent = 45  # Numeric (int)
        ds.Exposure = 13  # Numeric (int)
        ds.FilterType = 'LARGE'  # String (SH)
        ds.GeneratorPower = 5  # Numeric (int)
        ds.FocalSpots = [1.6, 1.4]  # List of floats
        ds.ConvolutionKernel = 'FC19'  # String (SH)
        ds.PatientPosition = 'HFS'  # String (CS)
        ds.AcquisitionType = 'SPIRAL'  # String (CS)
        ds.RevolutionTime = 0.5  # Numeric (float)
        ds.SingleCollimationWidth = 2.0  # Numeric (float)
        ds.TotalCollimationWidth = 32.0  # Numeric (float)
        ds.TableFeedPerRotation = -2.6  # Numeric (float)
        ds.SpiralPitchFactor = 0.081  # Numeric (float)
        ds.FluoroscopyFlag = 'NO'  # String (CS)
        ds.CTDIvol = 26.5  # Numeric (float)
    
        # Performed Procedure Step Information
        ds.PerformedProcedureStepStartDate = date_str
        ds.PerformedProcedureStepStartTime = '121656'
        ds.PerformedProcedureStepID = '18241'
    
        # Image Information
        ds.InstanceNumber = slice_index + 1  # Numeric
        ds.AcquisitionNumber = 1  # Numeric
        ds.ContentDate = date_str
        ds.ContentTime = time_str
        ds.AcquisitionDate = date_str
        ds.AcquisitionTime = '123249.35'
    
        # Image Position and Orientation
        if not hasattr(self, 'bbox_min') or self.bbox_min is None:
            self.report({'ERROR'}, "Bounding box minimum (bbox_min) not defined.")
            return {'CANCELLED'}
    
        ds.ImagePositionPatient = [
            float(self.bbox_min.x),
            float(self.bbox_min.y),
            float(self.bbox_min.z + (slice_index * voxel_size))
        ]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # List of floats
        ds.SliceLocation = float(self.bbox_min.z + (slice_index * voxel_size))  # Numeric (float)
        ds.PositionReferenceIndicator = 'XY'
        ds.PatientOrientation = ['L', 'P']
    
        # Stack and Temporal Information
        ds.StackID = '1_3800_00001'
        ds.InStackPositionNumber = slice_index + 1  # Numeric
        ds.TemporalPositionIndex = int(phase_info.get('temporal_position_index', '1'))  # Numeric
        ds.NumberOfTemporalPositions = int(total_phases)  # Numeric
    
        # Pixel Data Characteristics
        ds.SamplesPerPixel = 1  # Numeric
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows, ds.Columns = slice_data.shape
        ds.PixelSpacing = [float(voxel_size), float(voxel_size)]  # List of floats
        ds.BitsAllocated = 16  # Numeric
        ds.BitsStored = 16  # Numeric
        ds.HighBit = 15  # Numeric
        ds.PixelRepresentation = 1  # Numeric
        ds.RescaleIntercept = 0.0  # Numeric
        ds.RescaleSlope = 1.0  # Numeric
        ds.WindowCenter = 40  # Numeric
        ds.WindowWidth = 400  # Numeric
    
        # Set Pixel Data
        if noise_std_dev > 0:
            noise = np.random.normal(0, noise_std_dev, slice_data.shape)
            slice_data += noise
    
        slice_data = np.clip(slice_data, -1024, 3071)
        pixel_array = slice_data.astype(np.int16)
        ds.PixelData = pixel_array.tobytes()
    
        # Filename includes phase index and slice index
        filename = os.path.join(output_dir, f"CT_Phase_{phase_info.get('series_number')}_Slice_{slice_index+1:04d}.dcm")
    
        # Save the DICOM file
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(filename)
    
        self.report({'INFO'}, f"Saved DICOM slice {slice_index+1} to {filename}")
        return {'FINISHED'}




def generate_uniform_directions(num_directions):
    """
    Generates uniformly distributed directions over a sphere using the Fibonacci lattice.
    
    Parameters:
    - num_directions: int, number of directions to generate.
    
    Returns:
    - directions: list of mathutils.Vector, uniformly distributed directions.
    """
    directions = []
    phi = math.pi * (3. - math.sqrt(5.))  # Golden angle in radians

    for i in range(num_directions):
        y = 1 - (i / float(num_directions - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        directions.append(Vector((x, y, z)))

    return directions



# Helper function to create BVHTree from BMesh
def bvh_tree_from_bmesh(bm):
    import mathutils
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    vertices = [v.co.copy() for v in bm.verts]
    polygons = [[v.index for v in f.verts] for f in bm.faces]
    bvhtree = mathutils.bvhtree.BVHTree.FromPolygons(vertices, polygons)
    return bvhtree

# Operator to set default density
class SetDefaultDensityOperator(bpy.types.Operator):
    bl_idname = "object.set_default_density_operator"
    bl_label = "Set Default Density"
    bl_description = "Initialize selected mesh objects with a default density"

    default_density: FloatProperty(
        name="Default Density",
        default=0.0,
        description="Default density value to assign to selected mesh objects"
    )

    def execute(self, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        for obj in selected_meshes:
            obj.density = self.default_density

        self.report({'INFO'}, f"Set density of {len(selected_meshes)} object(s) to {self.default_density}")
        return {'FINISHED'}

# Operator to set default priority
class SetDefaultPriorityOperator(bpy.types.Operator):
    bl_idname = "object.set_default_priority_operator"
    bl_label = "Set Default Priority"
    bl_description = "Initialize selected mesh objects with a default priority"

    default_priority: IntProperty(
        name="Default Priority",
        default=0,
        description="Default priority value to assign to selected mesh objects"
    )

    def execute(self, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes:
            self.report({'WARNING'}, "No mesh objects selected")
            return {'CANCELLED'}

        for obj in selected_meshes:
            obj.priority = self.default_priority

        self.report({'INFO'}, f"Set priority of {len(selected_meshes)} object(s) to {self.default_priority}")
        return {'FINISHED'}

# Panel Class
class VoxelizerPanel(bpy.types.Panel):
    bl_label = "Voxelizer Extension"
    bl_idname = "VIEW3D_PT_voxelizer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Voxelizer'

    def draw(self, context):
        layout = self.layout
        obj = context.object
        scene = context.scene
        settings = scene.voxelizer_settings  # Access the settings

        # Display the 'density' and 'priority' properties when a mesh object is selected
        if obj and obj.type == 'MESH':
            layout.label(text="Object Properties")
            layout.prop(obj, 'density')
            layout.prop(obj, 'priority')
            layout.operator("object.set_default_density_operator", text="Set Default Density")
            layout.operator("object.set_default_priority_operator", text="Set Default Priority")
            layout.separator()

        # Voxelization settings
        layout.label(text="Voxelization Settings")
        layout.prop(settings, 'voxel_size')
        layout.prop(settings, 'enable_4d_export')

        if settings.enable_4d_export:
            layout.prop(settings, 'start_frame')
            layout.prop(settings, 'end_frame')

        layout.separator()
        layout.label(text="Artifact Simulation")

        # Noise Simulation
        layout.prop(settings, 'enable_noise')
        if settings.enable_noise:
            layout.prop(settings, 'noise_std_dev')

        # Metal Artifact Simulation
        layout.prop(settings, 'enable_metal_artifacts')
        if settings.enable_metal_artifacts:
            layout.prop(settings, 'metal_threshold')
            layout.prop(settings, 'streak_intensity')
            layout.prop(settings, 'num_angles')

        # Partial Volume Effect Simulation
        layout.prop(settings, 'enable_pve')
        if settings.enable_pve:
            layout.prop(settings, 'pve_sigma')

        # Ring Artifact Simulation
        layout.prop(settings, 'enable_ring_artifacts')
        if settings.enable_ring_artifacts:
            layout.prop(settings, 'ring_intensity')
            layout.prop(settings, 'ring_frequency')
        
        #layout.prop(settings, 'enable_beam_hardening')
        #if settings.enable_beam_hardening:
        #    layout.prop(settings, 'metal_threshold_bh')

        # Voxelization Method
        layout.separator()
        layout.label(text="Voxelization Method")
        layout.prop(settings, 'voxelization_method', expand=True)
        if settings.voxelization_method == 'ACCURATE':
            layout.prop(settings, 'samples_per_voxel')
        elif settings.voxelization_method == 'FAST':
            layout.prop(settings, 'num_rays_per_voxel')

        layout.separator()
        layout.operator("object.voxelize_operator")

# Registration
classes = (
    VoxelizerSettings,
    VoxelizeOperator,
    SetDefaultDensityOperator,
    SetDefaultPriorityOperator,
    VoxelizerPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.density = FloatProperty(
        name="Density",
        description="Density value for the object",
        default=1.0,
        min=-1000,
    )
    bpy.types.Object.priority = IntProperty(
        name="Priority",
        description="Priority for density assignment in overlapping regions",
        default=0,
    )
    bpy.types.Scene.voxelizer_settings = PointerProperty(type=VoxelizerSettings)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Object.density
    del bpy.types.Object.priority
    del bpy.types.Scene.voxelizer_settings

if __name__ == "__main__":
    register()
