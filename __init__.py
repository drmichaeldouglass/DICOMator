bl_info = {
    "name": "DICOMator",
    "author": "Michael Douglass",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > DICOMator",
    "description": "Converts mesh objects into DICOM CT files",
    "warning": "",
    "doc_url": "https://github.com/drmichaeldouglass/DICOMator",
    "category": "3D View",
}

# Standard library imports
import os
import math
import time
from datetime import datetime
from functools import partial

# Blender imports
import bpy
import bmesh
from mathutils import Vector
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    FloatProperty,
    BoolProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
    EnumProperty,
)

# Try to import numpy - required
try:
    import numpy as np
except ImportError:
    print("NumPy is required but not found. Using bundled wheel.")

# Try to import scikit-image for line drawing
try:
    from skimage.draw import line
except ImportError:
    print("scikit-image is required but not found. Using bundled wheel.")

# Try to import pydicom - required for DICOM export
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid
except ImportError:
    print("pydicom is required but not found. Using bundled wheel.")

# Constants
AIR_DENSITY = -1000.0  # HU value for air
DEFAULT_DENSITY = 0.0   # Default density for objects
MAX_HU_VALUE = 3071     # Maximum HU value
MIN_HU_VALUE = -1024    # Minimum HU value

# Property Group for settings
class VoxelizerSettings(PropertyGroup):
    """Settings for the DICOMator addon."""
    
    # Patient information
    patient_name: StringProperty(
        name="Patient Name",
        description="Name of the patient",
        default="Anonymous"
    )
    
    patient_id: StringProperty(
        name="Patient ID",
        description="Patient identification number",
        default="12345678"
    )
    
    patient_sex: EnumProperty(
        name="Patient Sex",
        description="Sex of the patient",
        items=[
            ('M', "Male", "Male"),
            ('F', "Female", "Female"),
            ('O', "Other", "Other")
        ],
        default='M'
    )
    
    # Voxelization settings
    voxel_size: FloatProperty(
        name="Voxel Size (mm)",
        default=2.0,
        min=0.01,
        description="Size of each voxel in millimeters"
    )
    
    sampling_density: IntProperty(
        name="Sampling Density",
        default=5,
        min=1,
        max=10,
        description="Number of samples per voxel (higher = more accurate but slower)"
    )

    # 4D CT export settings
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
        default=10,
        min=1,
        description="End frame for 4D export"
    )
    
    # Artifact simulation settings
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
    
    # Ring artifact settings
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

    output_directory: StringProperty(
        name="Output Directory",
        description="Directory to save DICOM files",
        default="//DICOM_Output",
        subtype='DIR_PATH'
    )


# Operator Class
class VoxelizeOperator(Operator):
    """Voxelize selected meshes and export as DICOM CT series."""
    bl_idname = "object.voxelize_operator"
    bl_label = "Voxelize Selected Objects"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}  # Add BLOCKING option
    
    # Internal attributes for UIDs
    study_instance_uid: StringProperty()
    frame_of_reference_uid: StringProperty()
    
    # Progress tracking properties
    progress: FloatProperty(default=0.0, min=0.0, max=100.0)
    progress_message: StringProperty(default="Initializing...")
    _timer = None
    _processing_done = False
    _voxel_grid = None
    _context = None
    _selected_meshes = None
    _output_dir = None
    _phase_info = None
    _current_frame_idx = 0
    _total_frames = 1
    _processing_step = 'INIT'  # INIT, VOXELIZING, ARTIFACTS, EXPORTING, DONE
    
    def execute(self, context):
        """Initialize the modal operation."""
        self.progress = 0.0
        self.progress_message = "Starting voxelization..."
        self._processing_done = False
        self._context = context
        self._processing_step = 'INIT'
        
        # Get selected meshes
        self._selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not self._selected_meshes:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}
        
        # Create output directory
        settings = context.scene.voxelizer_settings
        output_dir = bpy.path.abspath(settings.output_directory)
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        
        # Generate UIDs
        self.study_instance_uid = generate_uid()
        self.frame_of_reference_uid = generate_uid()
        
        # Initialize 4D settings
        self._current_frame_idx = 0
        if settings.enable_4d_export:
            if settings.start_frame > settings.end_frame:
                self.report({'ERROR'}, "Start Frame must be less than or equal to End Frame")
                return {'CANCELLED'}
            self._total_frames = settings.end_frame - settings.start_frame + 1
        else:
            self._total_frames = 1
        
        # Start the timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        # Show a progress bar in the status bar
        context.workspace.status_text_set("DICOMator: Initializing...")
        
        return {'RUNNING_MODAL'}
    
    def modal(self, context, event):
        """Handle modal updates and processing steps."""
        if event.type == 'ESC':
            self.cancel(context)
            return {'CANCELLED'}
            
        if event.type == 'TIMER':
            # Process the current step
            if self._processing_step == 'INIT':
                # Initialize phase
                self.progress = 0.0
                settings = context.scene.voxelizer_settings
                
                if settings.enable_4d_export:
                    # Set up the current frame in 4D sequence
                    frame = settings.start_frame + self._current_frame_idx
                    scene = context.scene
                    scene.frame_set(frame)
                    
                    # Calculate percentage through motion cycle
                    percentage = (self._current_frame_idx + 1) / self._total_frames * 100
                    
                    # Set up phase info for this frame
                    self._phase_info = {
                        'series_description': f"Phase {self._current_frame_idx + 1} - {percentage:.1f}% of breathing cycle",
                        'series_instance_uid': generate_uid(),
                        'series_number': str(self._current_frame_idx + 1),
                        'temporal_position_index': str(frame),
                    }
                else:
                    # Single frame export
                    self._phase_info = {
                        'series_description': "Static CT Scan",
                        'series_instance_uid': generate_uid(),
                        'series_number': "1",
                        'temporal_position_index': "1",
                    }
                
                self.progress_message = "Starting voxelization..."
                self._processing_step = 'VOXELIZING'
            
            elif self._processing_step == 'VOXELIZING':
                # Voxelizing phase
                settings = context.scene.voxelizer_settings
                self.progress_message = "Voxelizing objects..."
                
                # Process voxelization
                self._voxel_grid = self.adaptive_sampling_voxelization(
                    self._selected_meshes,
                    settings.voxel_size,
                    min_samples=2,
                    max_samples=settings.sampling_density
                )
                
                self.progress = 30.0  # Voxelization is 30% of total progress
                self._processing_step = 'ARTIFACTS'
            
            elif self._processing_step == 'ARTIFACTS':
                # Apply artifacts
                settings = context.scene.voxelizer_settings
                self.progress_message = "Applying artifacts and effects..."
                
                # Apply artifact simulations
                self._voxel_grid = self.apply_artifacts(self._voxel_grid, settings)
                
                self.progress = 40.0  # Artifacts application is 10% of total progress
                self._processing_step = 'EXPORTING'
            
            elif self._processing_step == 'EXPORTING':
                # Export to DICOM
                settings = context.scene.voxelizer_settings
                self.progress_message = f"Exporting DICOM files for phase {self._current_frame_idx + 1}/{self._total_frames}..."
                
                # Export to DICOM
                result = self.export_to_dicom(
                    voxel_grid=self._voxel_grid,
                    voxel_size=settings.voxel_size,
                    output_dir=self._output_dir,
                    phase_info=self._phase_info,
                    total_phases=self._total_frames,
                    patient_name=settings.patient_name,
                    patient_id=settings.patient_id,
                    patient_sex=settings.patient_sex,
                    progress_callback=self.update_export_progress
                )
                
                if result != {'FINISHED'}:
                    self.cancel(context)
                    return {'CANCELLED'}
                
                # Progress calculation - exporting is 60% of total progress
                # 40% baseline + progress for frames completed
                frame_progress = 60.0 * (self._current_frame_idx + 1) / self._total_frames
                self.progress = 40.0 + frame_progress
                
                # Move to next frame or finish
                self._current_frame_idx += 1
                if self._current_frame_idx < self._total_frames:
                    self._processing_step = 'INIT'  # Start next frame
                else:
                    self._processing_step = 'DONE'
            
            elif self._processing_step == 'DONE':
                # Finished all processing
                self.progress = 100.0
                self.progress_message = "Voxelization complete!"
                
                # Restore original frame if necessary
                if context.scene.voxelizer_settings.enable_4d_export:
                    context.scene.frame_set(context.scene.frame_current)
                
                self._processing_done = True
                self.finish(context)
                return {'FINISHED'}
            
            # Update the status text with progress
            context.workspace.status_text_set(f"DICOMator: {self.progress_message} ({self.progress:.1f}%)")
            
            # Force a UI redraw to show progress
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        
        return {'RUNNING_MODAL'}
    
    def update_export_progress(self, slice_index, total_slices):
        """Update progress during DICOM export."""
        slice_progress = slice_index / total_slices
        self.progress = 40.0 + 60.0 * (self._current_frame_idx + slice_progress) / self._total_frames
    
    def finish(self, context):
        """Clean up and finish the operator."""
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
        
        context.workspace.status_text_set(None)  # Clear the status text
        self.report({'INFO'}, f"Voxelization complete. Output saved to {self._output_dir}")
    
    def cancel(self, context):
        """Cancel the operation and clean up."""
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
        
        context.workspace.status_text_set(None)  # Clear the status text
        self.report({'INFO'}, "Voxelization cancelled")
    
    def _get_point_status(self, point_vector, mesh_data, num_rays):
        """Helper to determine if a point is inside any mesh and its density."""
        point_meshes = []
        for data in mesh_data:
            if is_point_inside_mesh(point_vector, data['bvhtree'], num_rays=num_rays):
                point_meshes.append({
                    'density': data['density'],
                    'priority': data['priority']
                })
        
        if point_meshes:
            point_meshes.sort(key=lambda x: x['priority'], reverse=True)
            return True, point_meshes[0]['density']  # is_inside, density
        else:
            return False, AIR_DENSITY  # is_inside, density

    def process_single_frame(self, context, selected_meshes, output_dir):
        """Process a single frame for CT export."""
        settings = context.scene.voxelizer_settings
        scene = context.scene
        frame = scene.frame_current
        
        # Create phase info for single frame
        phase_info = {
            'series_description': "Static CT Scan",
            'series_instance_uid': generate_uid(),
            'series_number': "1",
            'temporal_position_index': "1",
        }
        
        # Voxelize meshes using adaptive sampling
        voxel_grid = self.adaptive_sampling_voxelization(
            selected_meshes,
            settings.voxel_size,
            min_samples=2,
            max_samples=settings.sampling_density
        )
        
        # Apply artifact simulations if enabled
        voxel_grid = self.apply_artifacts(voxel_grid, settings)
        
        # Export to DICOM
        return self.export_to_dicom(
            voxel_grid=voxel_grid,
            voxel_size=settings.voxel_size,
            output_dir=output_dir,
            phase_info=phase_info,
            total_phases=1,
            patient_name=settings.patient_name,
            patient_id=settings.patient_id,
            patient_sex=settings.patient_sex
        )
    
    def process_4d_ct(self, context, selected_meshes, output_dir):
        """Process multiple frames for 4D CT export."""
        settings = context.scene.voxelizer_settings
        scene = context.scene
        current_frame = scene.frame_current  # Store current frame to restore later
        
        # Validate frame range
        if settings.start_frame > settings.end_frame:
            self.report({'ERROR'}, "Start Frame must be less than or equal to End Frame")
            return {'CANCELLED'}
            
        total_frames = settings.end_frame - settings.start_frame + 1
        
        try:
            for frame_index, frame in enumerate(range(settings.start_frame, settings.end_frame + 1), start=1):
                # Set the scene to the current frame
                scene.frame_set(frame)
                
                # Calculate percentage through the motion cycle
                percentage = (frame_index / total_frames) * 100
                
                # Create phase info
                phase_info = {
                    'series_description': f"Phase {frame_index} - {percentage:.1f}% of breathing cycle",
                    'series_instance_uid': generate_uid(),
                    'series_number': str(frame_index),
                    'temporal_position_index': str(frame),
                }
                
                # Voxelize meshes for this phase
                voxel_grid = self.adaptive_sampling_voxelization(
                    selected_meshes,
                    settings.voxel_size,
                    min_samples=2,
                    max_samples=settings.sampling_density
                )
                
                # Apply artifact simulations if enabled
                voxel_grid = self.apply_artifacts(voxel_grid, settings)
                
                # Export to DICOM
                result = self.export_to_dicom(
                    voxel_grid=voxel_grid,
                    voxel_size=settings.voxel_size,
                    output_dir=output_dir,
                    phase_info=phase_info,
                    total_phases=total_frames,
                    patient_name=settings.patient_name,
                    patient_id=settings.patient_id,
                    patient_sex=settings.patient_sex
                )
                
                if result != {'FINISHED'}:
                    return result
                    
                self.report({'INFO'}, f"Exported phase {frame_index}/{total_frames}")
                
            return {'FINISHED'}
                
        finally:
            # Restore the original frame
            scene.frame_set(current_frame)
    
    def apply_artifacts(self, voxel_grid, settings):
        """Apply all enabled artifact simulations to the voxel grid."""
        # Apply noise if enabled
        if settings.enable_noise:
            noise_std_dev = settings.noise_std_dev
            self.report({'INFO'}, f"Adding noise with standard deviation {noise_std_dev}")
            noise = np.random.normal(0, noise_std_dev, voxel_grid.shape)
            voxel_grid += noise
        
        # Apply partial volume effect if enabled
        if settings.enable_pve:
            voxel_grid = self.apply_partial_volume_effect(voxel_grid, sigma=settings.pve_sigma)
            
        # Apply metal artifacts if enabled
        if settings.enable_metal_artifacts:
            self.simulate_radial_streaks(
                voxel_grid,
                metal_threshold=settings.metal_threshold,
                streak_intensity=settings.streak_intensity,
                num_angles=settings.num_angles
            )
            
        # Apply ring artifacts if enabled
        if settings.enable_ring_artifacts:
            self.simulate_ring_artifacts(
                voxel_grid,
                intensity=settings.ring_intensity,
                frequency=settings.ring_frequency
            )
            
        return voxel_grid

    def calculate_combined_bounding_box(self, meshes):
        """
        Calculate the combined bounding box of all selected meshes.
        
        Args:
            meshes: List of Blender mesh objects.
            
        Returns:
            Tuple of (bbox_min, bbox_max) Vectors representing the bounding box.
        """
        if not meshes:
            return Vector((0, 0, 0)), Vector((0, 0, 0))
        
        # Initialize with the first mesh's bounding box
        first_mesh = meshes[0]
        bbox_min, bbox_max = first_mesh.bound_box[0], first_mesh.bound_box[6]
        bbox_min = first_mesh.matrix_world @ Vector(bbox_min)
        bbox_max = first_mesh.matrix_world @ Vector(bbox_max)
        
        # Expand bounding box to include all meshes
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

    def voxelize_meshes(self, meshes, voxel_size, samples_per_dimension=5):
        """
        Voxelize the selected meshes using an optimized sampling approach.
        
        Args:
            meshes: List of Blender mesh objects
            voxel_size: Size of each voxel in world units
            samples_per_dimension: Number of samples along each dimension per voxel
            
        Returns:
            3D numpy array representing the voxel grid
        """
        # Calculate the combined bounding box
        bbox_min, bbox_max = self.calculate_combined_bounding_box(meshes)
        self.bbox_min = bbox_min  # Store for DICOM export
        self.bbox_max = bbox_max
        
        # Calculate grid dimensions
        grid_size = bbox_max - bbox_min
        grid_dims = tuple(int(np.ceil(s / voxel_size)) for s in grid_size)
        
        # Initialize voxel grid with air density
        voxel_grid = np.full(grid_dims, AIR_DENSITY, dtype=np.float32)
        
        # Prepare BVH trees and object properties for all meshes
        mesh_data = []
        for mesh in meshes:
            # Get density and priority attributes
            density = getattr(mesh, 'density', DEFAULT_DENSITY)
            priority = getattr(mesh, 'priority', 0)
            
            # Create BMesh and BVHTree
            bm = bmesh.new()
            bm.from_mesh(mesh.data)
            bm.transform(mesh.matrix_world)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            
            # Create BVHTree
            bvhtree = bvh_tree_from_bmesh(bm)
            
            # Store mesh data
            mesh_data.append({
                'bvhtree': bvhtree,
                'density': density,
                'priority': priority,
                'bmesh': bm  # Store for cleanup later
            })
        
        # Generate sample points within each voxel
        sample_offsets = np.linspace(0, voxel_size, samples_per_dimension, endpoint=False) 
        sample_offsets += (voxel_size / (2 * samples_per_dimension))  # Center samples
        
        # Create 3D grid of sample points
        sample_points = np.array(
            np.meshgrid(sample_offsets, sample_offsets, sample_offsets, indexing='ij')
        ).reshape(3, -1).T
        
        total_samples = len(sample_points)
        
        # Process voxels in chunks for better memory usage
        chunk_size = 1000  # Adjust based on available memory
        total_voxels = np.prod(grid_dims)
        indices = np.array(list(np.ndindex(grid_dims)))
        
        # Report total voxels
        self.report({'INFO'}, f"Processing {total_voxels} voxels with {total_samples} samples per voxel")
        
        # Process voxels in chunks
        for chunk_start in range(0, total_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_voxels)
            chunk_indices = indices[chunk_start:chunk_end]
            
            # Process this chunk of voxels
            for idx in chunk_indices:
                # Calculate voxel world position
                voxel_min = bbox_min + Vector((
                    idx[0] * voxel_size,
                    idx[1] * voxel_size,
                    idx[2] * voxel_size
                ))
                
                # Convert to NumPy array for faster operations
                voxel_min_np = np.array(voxel_min)
                
                # Calculate all sample points in world coordinates for this voxel
                voxel_samples = voxel_min_np + sample_points
                
                # Initialize sample results
                sample_results = []
                
                # For each sample point, check which meshes contain it
                for sample in voxel_samples:
                    sample_point = Vector(sample)
                    point_meshes = []
                    
                    # Check if point is inside any mesh
                    for mesh_idx, data in enumerate(mesh_data):
                        if is_point_inside_mesh(sample_point, data['bvhtree']):
                            point_meshes.append({
                                'density': data['density'],
                                'priority': data['priority']
                            })
                    
                    if point_meshes:
                        # Sort by priority (highest first) and use highest priority mesh
                        point_meshes.sort(key=lambda x: x['priority'], reverse=True)
                        sample_results.append(point_meshes[0]['density'])
                    else:
                        sample_results.append(AIR_DENSITY)
                
                # Calculate voxel density based on sample results
                if sample_results:
                    # Take the average of sample densities
                    voxel_grid[tuple(idx)] = np.mean(sample_results)
            
            # Report progress
            progress = chunk_end / total_voxels * 100
            self.report({'INFO'}, f"Voxelization progress: {progress:.1f}%")
        
        # Clean up BMeshes
        for data in mesh_data:
            data['bmesh'].free()
        
        return voxel_grid

    def adaptive_sampling_voxelization(self, meshes, voxel_size, min_samples=2, max_samples=8):
        """
        Voxelize meshes using adaptive sampling - more samples near surfaces, fewer in uniform regions.
        
        Args:
            meshes: List of Blender mesh objects
            voxel_size: Size of each voxel in world units
            min_samples: Minimum samples per dimension in uniform regions
            max_samples: Maximum samples per dimension near boundaries
            
        Returns:
            3D numpy array representing the voxel grid
        """
        # Calculate the combined bounding box
        bbox_min, bbox_max = self.calculate_combined_bounding_box(meshes)
        self.bbox_min = bbox_min  # Store for DICOM export
        self.bbox_max = bbox_max
        
        bbox_min_np = np.array(bbox_min) # NumPy version for faster math
        
        # Calculate grid dimensions
        grid_size = bbox_max - bbox_min
        grid_dims = tuple(int(np.ceil(s / voxel_size)) for s in grid_size)
        
        # Initialize voxel grid with air density
        voxel_grid = np.full(grid_dims, AIR_DENSITY, dtype=np.float32)
        # Initialize boundary mask
        boundary_mask = np.zeros(grid_dims, dtype=bool)
        
        # Temporary grid to store (is_inside, density) for each voxel's center point
        voxel_info_dtype = [('is_inside', bool), ('density', np.float32)]
        voxel_info_grid = np.empty(grid_dims, dtype=voxel_info_dtype)
        voxel_info_grid['is_inside'] = False # Default: not inside
        voxel_info_grid['density'] = AIR_DENSITY # Default: air density
        
        # Prepare BVH trees and object properties for all meshes
        mesh_data = []
        for mesh in meshes:
            # Get density and priority attributes
            density = getattr(mesh, 'density', DEFAULT_DENSITY)
            priority = getattr(mesh, 'priority', 0)
            
            # Create BMesh and BVHTree
            bm = bmesh.new()
            bm.from_mesh(mesh.data)
            bm.transform(mesh.matrix_world)
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            
            # Create BVHTree
            bvhtree = bvh_tree_from_bmesh(bm)
            
            # Store mesh data
            mesh_data.append({
                'bvhtree': bvhtree,
                'density': density,
                'priority': priority,
                'bmesh': bm  # Store for cleanup later
            })
        
        # First pass:
        self.report({'INFO'}, "First pass: Evaluating voxel centers...")
        
        total_voxels = np.prod(grid_dims)
        indices = np.array(list(np.ndindex(grid_dims))) # Array of index tuples
        chunk_size = 10000  # Increased chunk size for potentially faster processing with fewer overheads

        # Precompute center offset for voxels (center of the voxel)
        voxel_center_offset_np = np.array([0.5, 0.5, 0.5]) * voxel_size

        # Sub-pass 1.1: Evaluate all voxel centers
        for chunk_start in range(0, total_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_voxels)
            current_chunk_indices = indices[chunk_start:chunk_end]

            for idx_arr in current_chunk_indices: # idx_arr is a numpy array [x,y,z]
                idx_tuple = tuple(idx_arr) # Use tuple for indexing numpy arrays
                
                # Calculate voxel center world position
                voxel_origin_np = bbox_min_np + idx_arr * voxel_size
                center_point_np = voxel_origin_np + voxel_center_offset_np
                center_vector = Vector(center_point_np.tolist()) # Convert to Blender Vector

                is_inside, density = self._get_point_status(center_vector, mesh_data, num_rays=3)
                
                voxel_grid[idx_tuple] = density
                voxel_info_grid[idx_tuple] = (is_inside, density)

            progress = chunk_end / total_voxels * 100
            self.report({'INFO'}, f"First pass (center eval) progress: {progress:.1f}%")

        self.report({'INFO'}, "First pass: Detecting boundary regions...")

        # Define neighbor index offsets (for 6-connectivity)
        neighbor_index_offsets = np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ], dtype=int)

        # Sub-pass 1.2: Detect boundaries using pre-calculated voxel_info_grid
        for chunk_start in range(0, total_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_voxels)
            current_chunk_indices = indices[chunk_start:chunk_end]

            for idx_arr in current_chunk_indices:
                idx_tuple = tuple(idx_arr)
                center_is_inside = voxel_info_grid[idx_tuple]['is_inside']
                center_density = voxel_info_grid[idx_tuple]['density']
                
                is_boundary = False
                for offset_arr in neighbor_index_offsets:
                    neighbor_idx_arr = idx_arr + offset_arr
                    neighbor_idx_tuple = tuple(neighbor_idx_arr)

                    # Check if neighbor is within grid bounds
                    if all(0 <= c < dim for c, dim in zip(neighbor_idx_arr, grid_dims)):
                        neighbor_is_inside = voxel_info_grid[neighbor_idx_tuple]['is_inside']
                        neighbor_density = voxel_info_grid[neighbor_idx_tuple]['density']
                    else:
                        # Neighbor is outside the grid, treat as air
                        neighbor_is_inside = False
                        neighbor_density = AIR_DENSITY
                    
                    # Boundary condition: different inside/outside status or significant density change
                    if (center_is_inside != neighbor_is_inside or 
                        abs(center_density - neighbor_density) > 10):  # Threshold for material difference
                        is_boundary = True
                        break  # Found a boundary, no need to check other neighbors
                
                if is_boundary:
                    boundary_mask[idx_tuple] = True
            
            progress = chunk_end / total_voxels * 100
            self.report({'INFO'}, f"First pass (boundary detection) progress: {progress:.1f}%")
        
        # Second pass: Apply adaptive sampling only to boundary regions
        self.report({'INFO'}, "Second pass: Applying adaptive sampling to boundary voxels...")
        
        # Get boundary voxel indices
        boundary_indices = np.argwhere(boundary_mask)
        total_boundary_voxels = len(boundary_indices)
        
        self.report({'INFO'}, f"Found {total_boundary_voxels} boundary voxels out of {total_voxels} total voxels")
        
        if total_boundary_voxels == 0:
            self.report({'INFO'}, "No boundary voxels found, skipping second pass")
            # Clean up BMeshes
            for data in mesh_data:
                data['bmesh'].free()
            return voxel_grid
        
        # Generate high-resolution sample offsets for boundary regions
        max_sample_offsets = np.linspace(0, voxel_size, max_samples, endpoint=False)
        max_sample_offsets += voxel_size / (2 * max_samples)  # Center samples
        
        max_sample_points = np.array(
            np.meshgrid(max_sample_offsets, max_sample_offsets, max_sample_offsets, indexing='ij')
        ).reshape(3, -1).T
        
        # Process boundary voxels with high-resolution sampling
        for i, idx in enumerate(boundary_indices):
            idx = tuple(idx)
            
            # Calculate voxel world position
            voxel_min = bbox_min + Vector((
                idx[0] * voxel_size,
                idx[1] * voxel_size,
                idx[2] * voxel_size
            ))
            
            # Convert to NumPy array for faster operations
            voxel_min_np = np.array(voxel_min)
            
            # Calculate all sample points in world coordinates for this voxel
            voxel_samples = voxel_min_np + max_sample_points
            
            # Initialize sample results
            sample_results = []
            
            # For each sample point, check which meshes contain it
            for sample in voxel_samples:
                sample_point = Vector(sample.tolist()) # Ensure sample is list for Vector constructor
                
                # Use the _get_point_status helper here as well for consistency
                # Though the original logic for point_meshes is fine, this standardizes
                is_inside_sample, density_sample = self._get_point_status(sample_point, mesh_data, num_rays=5) # Default num_rays for higher accuracy
                sample_results.append(density_sample)
            
            # Calculate voxel density based on sample results
            if sample_results:
                # Take the average of sample densities
                voxel_grid[idx] = np.mean(sample_results)
            
            # Report progress periodically
            if (i + 1) % 100 == 0 or i == total_boundary_voxels - 1:
                progress = (i + 1) / total_boundary_voxels * 100
                self.report({'INFO'}, f"Second pass progress: {progress:.1f}%")
        
        # Clean up BMeshes
        for data in mesh_data:
            data['bmesh'].free()
        
        return voxel_grid

    def simulate_radial_streaks(self, voxel_grid, metal_threshold, streak_intensity, num_angles):
        """
        Simulate metal artifacts as radial streaks.
        
        Args:
            voxel_grid: 3D numpy array of voxel data
            metal_threshold: Density threshold to consider voxels as metal
            streak_intensity: Intensity of the streaks
            num_angles: Number of angles for streak rays
        """
        # Process each slice separately
        for z in range(voxel_grid.shape[2]):
            # Get slice data
            slice_data = voxel_grid[:, :, z].copy()
            
            # Find metal points
            metal_indices = np.argwhere(slice_data >= metal_threshold)
            if metal_indices.size == 0:
                continue
            
            # Get slice dimensions
            height, width = slice_data.shape
            
            # Process each metal point
            for metal_voxel in metal_indices:
                x0, y0 = metal_voxel
                
                # Create streaks in all directions
                for angle in np.linspace(0, 2 * np.pi, num=num_angles, endpoint=False):
                    # Calculate end point
                    x1 = int(x0 + height * np.sin(angle))
                    y1 = int(y0 + width * np.cos(angle))
                    
                    # Ensure points are within bounds
                    x1 = np.clip(x1, 0, height - 1)
                    y1 = np.clip(y1, 0, width - 1)
                    
                    # Draw line from metal to edge
                    rr, cc = line(x0, y0, x1, y1)
                    
                    # Clip coordinates
                    rr = np.clip(rr, 0, height - 1)
                    cc = np.clip(cc, 0, width - 1)
                    
                    # Calculate distance-based attenuation
                    distances = np.sqrt((rr - x0) ** 2 + (cc - y0) ** 2)
                    attenuation = np.exp(-distances / (0.1 * max(height, width)))
                    
                    # Apply streak intensity with attenuation
                    slice_data[rr, cc] += streak_intensity * attenuation
            
            # Clip values to valid HU range
            slice_data = np.clip(slice_data, MIN_HU_VALUE, MAX_HU_VALUE)
            
            # Update slice in voxel grid
            voxel_grid[:, :, z] = slice_data

    def simulate_ring_artifacts(self, voxel_grid, intensity, frequency):
        """
        Simulate ring artifacts in CT images.
        
        Args:
            voxel_grid: 3D numpy array of voxel data
            intensity: Intensity of the rings
            frequency: Frequency of the rings
        """
        # Calculate center of each slice
        center = (voxel_grid.shape[0] // 2, voxel_grid.shape[1] // 2)
        
        # Create coordinate grid
        y, x = np.ogrid[:voxel_grid.shape[0], :voxel_grid.shape[1]]
        
        # Process each slice
        for z in range(voxel_grid.shape[2]):
            # Get slice data
            slice_data = voxel_grid[:, :, z]
            
            # Calculate radial distance from center
            radius = np.hypot(x - center[1], y - center[0])
            
            # Create ring pattern
            rings = np.sin(2 * np.pi * frequency * radius / voxel_grid.shape[0])
            
            # Apply rings to slice
            slice_data += intensity * rings
            
            # Update slice in voxel grid
            voxel_grid[:, :, z] = slice_data

    def apply_partial_volume_effect(self, voxel_grid, sigma):
        """
        Apply Gaussian smoothing to simulate partial volume effects.
        
        Args:
            voxel_grid: 3D numpy array of voxel data
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Smoothed voxel grid
        """
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(voxel_grid, sigma=sigma)

    def export_to_dicom(
        self,
        voxel_grid,
        voxel_size,
        output_dir,
        phase_info,
        total_phases=1,
        patient_name="Anonymous",
        patient_id="12345678",
        patient_sex="M",
        progress_callback=None
    ):
        """
        Export the voxel grid as a series of DICOM slices.
        
        Args:
            voxel_grid: 3D numpy array of voxel data
            voxel_size: Size of each voxel in world units
            output_dir: Directory to save DICOM files
            phase_info: Dictionary with phase information
            total_phases: Total number of phases in 4D sequence
            patient_name: Name of the patient
            patient_id: Patient ID
            patient_sex: Patient sex (M/F/O)
            progress_callback: Optional callback function(slice_index, total_slices)
            
        Returns:
            Status dictionary
        """
        # Get number of slices
        num_slices = voxel_grid.shape[2]
        
        # Clip values to valid HU range
        voxel_grid = np.clip(voxel_grid, MIN_HU_VALUE, MAX_HU_VALUE)
        
        # Export each slice
        for i in range(num_slices):
            slice_data = voxel_grid[:, :, i]
            
            result = self.save_dicom_slice(
                slice_data=slice_data,
                voxel_size=voxel_size,
                slice_index=i,
                output_dir=output_dir,
                phase_info=phase_info,
                total_phases=total_phases,
                num_slices=num_slices,
                patient_name=patient_name,
                patient_id=patient_id,
                patient_sex=patient_sex
            )
            
            if result != {'FINISHED'}:
                return result
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, num_slices)
                
        return {'FINISHED'}

    def save_dicom_slice(
        self,
        slice_data,
        voxel_size,
        slice_index,
        output_dir,
        phase_info,
        total_phases=1,
        num_slices=1,
        patient_name="Anonymous",
        patient_id="12345678",
        patient_sex="M"
    ):
        """
        Save a single slice as a DICOM file.
        
        Args:
            slice_data: 2D numpy array of slice data
            voxel_size: Size of each voxel in world units
            slice_index: Index of the slice
            output_dir: Directory to save DICOM files
            phase_info: Dictionary with phase information
            total_phases: Total number of phases in 4D sequence
            num_slices: Total number of slices
            patient_name: Name of the patient
            patient_id: Patient ID
            patient_sex: Patient sex (M/F/O)
            
        Returns:
            Status dictionary
        """
        # Get current date and time
        current_datetime = datetime.now()
        date_str = current_datetime.strftime('%Y%m%d')
        time_str = current_datetime.strftime('%H%M%S.%f')
        
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
        
        # Patient information
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = ''
        ds.PatientSex = patient_sex
        
        # Study information
        ds.StudyInstanceUID = self.study_instance_uid
        ds.FrameOfReferenceUID = self.frame_of_reference_uid
        ds.StudyID = '1'
        ds.AccessionNumber = '1'
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.ReferringPhysicianName = ''
        
        # Series information
        ds.SeriesInstanceUID = phase_info.get('series_instance_uid', generate_uid())
        ds.SeriesNumber = int(phase_info.get('series_number', '1'))
        ds.SeriesDescription = phase_info.get('series_description', 'CT Series')
        ds.SeriesDate = date_str
        ds.SeriesTime = time_str
        
        # Equipment information
        ds.Modality = 'CT'
        ds.Manufacturer = 'DICOMator'
        ds.InstitutionName = 'Virtual Hospital'
        ds.StationName = 'Blender'
        
        # Image information
        ds.InstanceNumber = slice_index + 1
        ds.AcquisitionNumber = 1
        ds.ContentDate = date_str
        ds.ContentTime = time_str
        ds.AcquisitionDate = date_str
        ds.AcquisitionTime = time_str
        
        # Image position and orientation
        ds.ImagePositionPatient = [
            float(self.bbox_min.x),
            float(self.bbox_min.y),
            float(self.bbox_min.z + (slice_index * voxel_size))
        ]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.SliceLocation = float(self.bbox_min.z + (slice_index * voxel_size))
        
        # Temporal information for 4D
        if total_phases > 1:
            ds.TemporalPositionIndex = int(phase_info.get('temporal_position_index', '1'))
            ds.NumberOfTemporalPositions = total_phases
        
        # Pixel data characteristics
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows, ds.Columns = slice_data.shape
        ds.PixelSpacing = [float(voxel_size), float(voxel_size)]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # Signed
        ds.RescaleIntercept = 0.0
        ds.RescaleSlope = 1.0
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        
        # Convert to int16 for DICOM
        pixel_array = slice_data.astype(np.int16)
        ds.PixelData = pixel_array.tobytes()
        
        # Create filename
        filename = os.path.join(output_dir, f"CT_Phase_{phase_info.get('series_number')}_Slice_{slice_index+1:04d}.dcm")
        
        # Save the DICOM file
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        try:
            ds.save_as(filename)
            self.report({'INFO'}, f"Saved slice {slice_index+1}/{num_slices}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error saving DICOM file: {str(e)}")
            return {'CANCELLED'}


# Helper Functions

def is_point_inside_mesh(point, bvhtree, num_rays=5):
    """
    Determines if a point is inside a mesh using ray casting.
    
    Args:
        point: Vector, position to test
        bvhtree: BVHTree of the mesh
        num_rays: Number of rays to cast (more = more accurate)
        
    Returns:
        bool: True if the point is inside the mesh
    """
    # Generate uniformly distributed ray directions
    ray_directions = generate_uniform_directions(num_rays)
    
    inside_count = 0
    
    for direction in ray_directions:
        ray_origin = point.copy()
        ray_direction = direction.normalized()
        hit_count = 0
        
        # Cast ray and count intersections
        while True:
            hit = bvhtree.ray_cast(ray_origin, ray_direction)
            
            if hit[0] is None:  # No intersection
                break
                
            hit_count += 1
            
            # Move slightly past the intersection
            ray_origin = hit[0] + ray_direction * 1e-5
            
            # Avoid infinite loops
            if hit_count > 100:
                break
                
        # If odd number of hits, point is inside
        if hit_count % 2 == 1:
            inside_count += 1
    
    # Use majority vote
    return inside_count > (num_rays / 2)


def generate_uniform_directions(num_directions):
    """
    Generate uniformly distributed directions on a sphere.
    
    Args:
        num_directions: Number of directions to generate
        
    Returns:
        List of Vector objects representing directions
    """
    directions = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle
    
    for i in range(num_directions):
        y = 1.0 - (i / float(num_directions - 1)) * 2.0  # y goes from 1 to -1
        radius = math.sqrt(1.0 - y * y)  # Radius at y
        
        theta = phi * i  # Golden angle increment
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        directions.append(Vector((x, y, z)))
        
    return directions


def bvh_tree_from_bmesh(bm):
    """
    Create a BVHTree from a BMesh.
    
    Args:
        bm: BMesh object
        
    Returns:
        BVHTree object
    """
    import mathutils
    
    # Ensure lookup tables
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # Get vertices and polygons
    vertices = [v.co.copy() for v in bm.verts]
    polygons = [[v.index for v in f.verts] for f in bm.faces]
    
    # Create BVHTree
    return mathutils.bvhtree.BVHTree.FromPolygons(vertices, polygons)


def ray_traversal_voxelization(meshes, bbox_min, bbox_max, voxel_size):
    # Initialize grid
    grid_size = bbox_max - bbox_min
    grid_dims = tuple(int(np.ceil(s / voxel_size)) for s in grid_size)
    voxel_grid = np.full(grid_dims, AIR_DENSITY, dtype=np.float32)
    unprocessed_voxels = set(np.ndindex(grid_dims))
    
    # Prepare meshes (BVH trees, etc.)
    mesh_data = prepare_meshes(meshes)
    
    # Generate rays (origins and directions)
    rays = generate_rays(bbox_min, bbox_max)
    
    # Process rays
    for origin, direction in rays:
        # Process all mesh intersections along this ray
        intersections = []
        for mesh_idx, data in enumerate(mesh_data):
            hits = get_all_intersections(origin, direction, data['bvhtree'])
            for hit_point, hit_normal in hits:
                is_entry = direction.dot(hit_normal) < 0
                intersections.append({
                    'point': hit_point,
                    'is_entry': is_entry,
                    'mesh_idx': mesh_idx,
                    'distance': (hit_point - origin).length
                })
        
        # Sort intersections by distance
        intersections.sort(key=lambda x: x['distance'])
        
        # Track which meshes we're inside
        inside_meshes = {}
        current_point = origin
        
        # Walk along the ray
        for intersection in intersections:
            # Find voxels between current_point and intersection point
            voxels_to_process = voxels_between(current_point, intersection['point'], 
                                              bbox_min, voxel_size, unprocessed_voxels)
            
            # Process these voxels based on which meshes we're inside
            for voxel_idx in voxels_to_process:
                if inside_meshes:  # If we're inside at least one mesh
                    highest_priority = max(inside_meshes.values(), key=lambda x: x['priority'])
                    voxel_grid[voxel_idx] = highest_priority['density']
                    unprocessed_voxels.remove(voxel_idx)
            
            # Update which meshes we're inside
            mesh_idx = intersection['mesh_idx']
            if intersection['is_entry']:
                inside_meshes[mesh_idx] = mesh_data[mesh_idx]
            elif mesh_idx in inside_meshes:
                del inside_meshes[mesh_idx]
            
            current_point = intersection['point']
            
        # Continue with ray until we exit the bounding box
        if inside_meshes:
            exit_point = ray_bbox_intersection(current_point, direction, bbox_min, bbox_max)
            voxels_to_process = voxels_between(current_point, exit_point, 
                                              bbox_min, voxel_size, unprocessed_voxels)
            
            # Process these voxels
            for voxel_idx in voxels_to_process:
                highest_priority = max(inside_meshes.values(), key=lambda x: x['priority'])
                voxel_grid[voxel_idx] = highest_priority['density']
                unprocessed_voxels.remove(voxel_idx)
        
        # If no unprocessed voxels remain, we're done
        if not unprocessed_voxels:
            break
    
    return voxel_grid


def prepare_meshes(meshes):
    """
    Prepare mesh data for ray traversal voxelization.
    
    Args:
        meshes: List of Blender mesh objects
        
    Returns:
        List of dictionaries with prepared mesh data
    """
    mesh_data = []
    for mesh in meshes:
        # Get density and priority attributes
        density = getattr(mesh, 'density', DEFAULT_DENSITY)
        priority = getattr(mesh, 'priority', 0)
        
        # Create BMesh and BVHTree
        bm = bmesh.new()
        bm.from_mesh(mesh.data)
        bm.transform(mesh.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        
        # Create BVHTree
        bvhtree = bvh_tree_from_bmesh(bm)
        
        # Store mesh data
        mesh_data.append({
            'bvhtree': bvhtree,
            'density': density,
            'priority': priority,
            'bmesh': bm
        })
    
    return mesh_data


def generate_rays(bbox_min, bbox_max):
    """
    Generate rays for ray traversal voxelization.
    
    Args:
        bbox_min: Vector representing minimum corner of bounding box
        bbox_max: Vector representing maximum corner of bounding box
        
    Returns:
        List of (origin, direction) tuples for ray casting
    """
    rays = []
    
    # Calculate size of bounding box
    bbox_size = bbox_max - bbox_min
    max_dimension = max(bbox_size.x, bbox_size.y, bbox_size.z)
    
    # Determine number of rays based on bounding box size
    # More rays for larger objects to ensure adequate coverage
    num_rays_per_side = max(10, int(max_dimension / 5))
    
    # Generate rays from multiple sides of the bounding box
    # X axis rays
    for y in np.linspace(bbox_min.y, bbox_max.y, num_rays_per_side):
        for z in np.linspace(bbox_min.z, bbox_max.z, num_rays_per_side):
            # Ray from -X side
            origin = Vector((bbox_min.x - 1, y, z))
            direction = Vector((1, 0, 0))
            rays.append((origin, direction))
            
            # Ray from +X side
            origin = Vector((bbox_max.x + 1, y, z))
            direction = Vector((-1, 0, 0))
            rays.append((origin, direction))
    
    # Y axis rays
    for x in np.linspace(bbox_min.x, bbox_max.x, num_rays_per_side):
        for z in np.linspace(bbox_min.z, bbox_max.z, num_rays_per_side):
            # Ray from -Y side
            origin = Vector((x, bbox_min.y - 1, z))
            direction = Vector((0, 1, 0))
            rays.append((origin, direction))
            
            # Ray from +Y side
            origin = Vector((x, bbox_max.y + 1, z))
            direction = Vector((0, -1, 0))
            rays.append((origin, direction))
    
    # Z axis rays
    for x in np.linspace(bbox_min.x, bbox_max.x, num_rays_per_side):
        for y in np.linspace(bbox_min.y, bbox_max.y, num_rays_per_side):
            # Ray from -Z side
            origin = Vector((x, y, bbox_min.z - 1))
            direction = Vector((0, 0, 1))
            rays.append((origin, direction))
            
            # Ray from +Z side
            origin = Vector((x, y, bbox_max.z + 1))
            direction = Vector((0, 0, -1))
            rays.append((origin, direction))
    
    return rays


def get_all_intersections(origin, direction, bvhtree):
    """
    Get all intersections of a ray with a mesh.
    
    Args:
        origin: Vector, ray origin point
        direction: Vector, ray direction
        bvhtree: BVHTree object for the mesh
        
    Returns:
        List of (hit_point, hit_normal) tuples
    """
    intersections = []
    current_origin = origin.copy()
    
    # Maximum number of intersections to prevent infinite loops
    max_hits = 100
    hit_count = 0
    
    while hit_count < max_hits:
        # Cast ray from current origin
        hit = bvhtree.ray_cast(current_origin, direction)
        
        if hit[0] is None:  # No intersection
            break
            
        hit_point, hit_normal, _, _ = hit
        
        # Add intersection to list
        intersections.append((hit_point.copy(), hit_normal.copy()))
        
        # Move slightly past the intersection for next ray
        current_origin = hit_point + direction * 1e-5
        hit_count += 1
    
    return intersections


def voxels_between(start_point, end_point, bbox_min, voxel_size, unprocessed_voxels):
    """
    Get all voxels along a ray segment between two points.
    
    Args:
        start_point: Vector, start point of segment
        end_point: Vector, end point of segment
        bbox_min: Vector, minimum corner of bounding box
        voxel_size: Size of each voxel
        unprocessed_voxels: Set of unprocessed voxel indices
        
    Returns:
        List of voxel indices that intersect the segment
    """
    # Use 3D Digital Differential Analyzer (DDA) algorithm
    direction = end_point - start_point
    distance = direction.length
    
    if distance == 0:
        # Start and end points are the same
        idx = tuple(np.floor((start_point - bbox_min) / voxel_size).astype(int))
        if idx in unprocessed_voxels:
            return [idx]
        return []
    
    direction.normalize()
    
    # Determine step count based on length
    step_count = int(distance / (voxel_size * 0.5)) + 1
    step_size = distance / step_count
    
    voxels = []
    for i in range(step_count + 1):
        point = start_point + direction * (i * step_size)
        idx = tuple(np.floor((point - bbox_min) / voxel_size).astype(int))
        
        # Check if this voxel is in bounds and unprocessed
        if idx in unprocessed_voxels and idx not in voxels:
            voxels.append(idx)
    
    return voxels


def ray_bbox_intersection(origin, direction, bbox_min, bbox_max):
    """
    Find the point where a ray exits a bounding box.
    
    Args:
        origin: Vector, ray origin point
        direction: Vector, ray direction
        bbox_min: Vector, minimum corner of bounding box
        bbox_max: Vector, maximum corner of bounding box
        
    Returns:
        Vector, exit point or None if ray doesn't intersect
    """
    # Initialize with default values
    t_min = float('-inf')
    t_max = float('inf')
    
    # Check for division by zero
    for i in range(3):
        origin_val = origin[i]
        direction_val = direction[i]
        bbox_min_val = bbox_min[i]
        bbox_max_val = bbox_max[i]
        
        if abs(direction_val) < 1e-10:  # Nearly zero
            # Ray is parallel to this axis, check if within bounds
            if origin_val < bbox_min_val or origin_val > bbox_max_val:
                return None  # Ray doesn't intersect box
        else:
            # Calculate intersection parameters
            t1 = (bbox_min_val - origin_val) / direction_val
            t2 = (bbox_max_val - origin_val) / direction_val
            
            # Ensure t1 <= t2
            if t1 > t2:
                t1, t2 = t2, t1
                
            # Update bounds
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            
            if t_min > t_max:
                return None  # Ray doesn't intersect box
    
    # If origin is inside the box, use t_max (exit point)
    # Otherwise use t_min (entry point)
    if t_min > 0:
        return origin + direction * t_min
    else:
        return origin + direction * t_max


# Operator to set default density
class SetDefaultDensityOperator(bpy.types.Operator):
    """Initialize selected mesh objects with a default density value."""
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
    """Initialize selected mesh objects with a default priority value."""
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
    """Panel for DICOMator settings and operations."""
    bl_label = "DICOMator"
    bl_idname = "VIEW3D_PT_dicomator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'DICOMator'

    def draw(self, context):
        layout = self.layout
        obj = context.object
        scene = context.scene
        settings = scene.voxelizer_settings

        # Object properties section (visible when a mesh is selected)
        if obj and obj.type == 'MESH':
            box = layout.box()
            box.label(text="Object Properties")
            box.prop(obj, 'density')
            box.prop(obj, 'priority')
            
            # Operators for setting defaults
            row = box.row(align=True)
            row.operator("object.set_default_density_operator", text="Set Default Density")
            row.operator("object.set_default_priority_operator", text="Set Default Priority")

        # Voxelization settings
        box = layout.box()
        box.label(text="Voxelization Settings")
        box.prop(settings, 'voxel_size')
        box.prop(settings, 'sampling_density')
        
        # 4D export settings
        box.prop(settings, 'enable_4d_export')
        if settings.enable_4d_export:
            row = box.row(align=True)
            row.prop(settings, 'start_frame')
            row.prop(settings, 'end_frame')

        # Artifact simulation
        box = layout.box()
        box.label(text="Artifact Simulation")
        
        # Noise settings
        row = box.row()
        row.prop(settings, 'enable_noise')
        if settings.enable_noise:
            row.prop(settings, 'noise_std_dev')
        
        # Metal artifact settings
        row = box.row()
        row.prop(settings, 'enable_metal_artifacts')
        if settings.enable_metal_artifacts:
            col = box.column(align=True)
            col.prop(settings, 'metal_threshold')
            col.prop(settings, 'streak_intensity')
            col.prop(settings, 'num_angles')
        
        # Partial volume effect settings
        row = box.row()
        row.prop(settings, 'enable_pve')
        if settings.enable_pve:
            row.prop(settings, 'pve_sigma')
        
        # Ring artifact settings
        row = box.row()
        row.prop(settings, 'enable_ring_artifacts')
        if settings.enable_ring_artifacts:
            col = box.column(align=True)
            col.prop(settings, 'ring_intensity')
            col.prop(settings, 'ring_frequency')

        # Patient information
        box = layout.box()
        box.label(text="Patient Information")
        box.prop(settings, "patient_name")
        box.prop(settings, "patient_id")
        box.prop(settings, "patient_sex")

        # Output directory selection (add before the voxelize button)
        box = layout.box()
        box.label(text="Export Settings")
        box.prop(settings, "output_directory")

        # Voxelize button
        layout.separator()
        layout.operator("object.voxelize_operator", text="Voxelize Selected Objects")
        
        # Show warning if no objects selected
        if not context.selected_objects:
            layout.label(text="No objects selected", icon='ERROR')
            layout.label(text="Select mesh objects to voxelize")


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
        name="Density (HU)",
        description="Density value in Hounsfield Units",
        default=0.0,
        min=-1024.0,
        max=3071.0,
    )
    
    bpy.types.Object.priority = IntProperty(
        name="Priority",
        description="Priority for density assignment in overlapping regions (higher values take precedence)",
        default=0,
        min=0,
    )
    
    bpy.types.Scene.voxelizer_settings = PointerProperty(type=VoxelizerSettings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Object.density
    del bpy.types.Object.priority
    del bpy.types.Scene.voxelizer_settings

if __name__ == "__main__":
    register()
