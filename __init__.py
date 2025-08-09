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
from mathutils.bvhtree import BVHTree
import numpy as np

# Add these DICOM-specific imports
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM export functionality will be disabled.")

# Constants
AIR_DENSITY = -1000.0  # HU value for air
DEFAULT_DENSITY = 0.0   # Default density for objects
MAX_HU_VALUE = 3071     # Maximum HU value
MIN_HU_VALUE = -1024    # Minimum HU value

# Safely fetch numeric properties even when Blender returns _PropertyDeferred
def _get_float_prop(props, name: str, default: float) -> float:
    try:
        val = getattr(props, name)
        return float(val)
    except Exception:
        return float(default)

# Safely fetch string properties even when Blender returns _PropertyDeferred
def _get_str_prop(props, name: str, default: str) -> str:
    try:
        val = getattr(props, name)
        return str(val) if val is not None else str(default)
    except Exception:
        return str(default)

def is_point_inside_mesh(point, obj):
    """
    Check if a point is inside a mesh using closest point method.
    
    Args:
        point: Vector, point to test in world coordinates
        obj: Blender mesh object
        
    Returns:
        bool: True if point is inside mesh, False otherwise
    """
    # Convert point to object's local coordinate system
    local_point = obj.matrix_world.inverted() @ point
    
    # Get closest point on mesh surface
    _, closest, nor, _ = obj.closest_point_on_mesh(local_point)
    
    # Calculate direction from test point to closest surface point
    direction = closest - local_point
    
    # If the direction aligns with the surface normal, point is outside
    # If it opposes the normal, point is inside
    if direction.dot(nor) > 0:
        return False  # Point is outside
    else:
        return True   # Point is inside

def export_voxel_grid_to_dicom(
        voxel_grid,
        voxel_size,
        output_dir,
        bbox_min,
        patient_name="Anonymous",
        patient_id="12345678",
        patient_sex="M",
        series_description="CT Series from DICOMator",
        progress_callback=None,
        direct_hu=False,
        patient_position="HFS",  # NEW: allow choosing patient position (orientation)
    ):
        """
        Export the voxel grid (binary or HU) as a series of DICOM slices.
        If direct_hu=True, 'voxel_grid' is expected to contain HU values (int16).
        Otherwise, positive voxels are mapped to DEFAULT_DENSITY and 0 to AIR_DENSITY.

        patient_position: DICOM PatientPosition code (e.g., HFS, FFS, HFP, FFP, HFDR, HFDL, FFDR, FFDL)
        """
        if not PYDICOM_AVAILABLE:
            return {'error': 'pydicom not available'}
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current date and time
        current_datetime = datetime.now()
        date_str = current_datetime.strftime('%Y%m%d')
        time_str = current_datetime.strftime('%H%M%S.%f')
        
        # Dimensions and units
        num_slices = voxel_grid.shape[2]
        # Convert to HU grid
        if direct_hu:
            hu_grid = np.array(voxel_grid, dtype=np.int16, copy=False)
        else:
            hu_grid = np.where(voxel_grid > 0, DEFAULT_DENSITY, AIR_DENSITY).astype(np.int16, copy=False)
        # Clip values to valid HU range
        hu_grid = np.clip(hu_grid, MIN_HU_VALUE, MAX_HU_VALUE).astype(np.int16, copy=False)

        # DICOM expects millimeters for spatial tags
        voxel_size_mm = float(voxel_size) * 1000.0
        bbox_min_mm = Vector((bbox_min.x * 1000.0, bbox_min.y * 1000.0, bbox_min.z * 1000.0))
        
        # Generate UIDs for this study
        study_instance_uid = generate_uid()
        frame_of_reference_uid = generate_uid()
        series_instance_uid = generate_uid()
        
        # Common DICOM metadata that doesn't change per slice
        common_metadata = {
            'study_instance_uid': study_instance_uid,
            'frame_of_reference_uid': frame_of_reference_uid,
            'series_instance_uid': series_instance_uid,
            'series_number': 1,
            'series_description': series_description,
            'patient_name': patient_name,
            'patient_id': patient_id,
            'patient_sex': patient_sex,
            'date_str': date_str,
            'time_str': time_str,
        }
        
        for i in range(num_slices):
            # Note: hu_grid is shaped (width, height, depth). For DICOM, Rows = Y (height), Columns = X (width)
            slice_data = hu_grid[:, :, i]
            try:
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
                ds.PatientName = common_metadata['patient_name']
                ds.PatientID = common_metadata['patient_id']
                ds.PatientBirthDate = ''
                ds.PatientSex = common_metadata['patient_sex']
                ds.PatientPosition = str(patient_position)  # CHANGED: use selected patient position
                
                # Study information
                ds.StudyInstanceUID = common_metadata['study_instance_uid']
                ds.FrameOfReferenceUID = common_metadata['frame_of_reference_uid']
                ds.StudyID = '1'
                ds.AccessionNumber = '1'
                ds.StudyDate = common_metadata['date_str']
                ds.StudyTime = common_metadata['time_str']
                ds.ReferringPhysicianName = ''
                
                # Series information
                ds.SeriesInstanceUID = common_metadata['series_instance_uid']
                ds.SeriesNumber = common_metadata['series_number']
                ds.SeriesDescription = common_metadata['series_description']
                ds.SeriesDate = common_metadata['date_str']
                ds.SeriesTime = common_metadata['time_str']
                
                # Equipment information
                ds.Modality = 'CT'
                ds.Manufacturer = 'DICOMator'
                ds.InstitutionName = 'Virtual Hospital'
                ds.StationName = 'Blender'
                
                # Image information
                ds.InstanceNumber = i + 1
                ds.AcquisitionNumber = 1
                ds.ContentDate = common_metadata['date_str']
                ds.ContentTime = common_metadata['time_str']
                ds.AcquisitionDate = common_metadata['date_str']
                ds.AcquisitionTime = common_metadata['time_str']
                
                # Image position and orientation
                ds.ImagePositionPatient = [
                    float(bbox_min_mm.x),
                    float(bbox_min_mm.y),
                    float(bbox_min_mm.z + (i * voxel_size_mm))
                ]
                ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # axial identity; patient position set above
                ds.SliceLocation = float(bbox_min_mm.z + (i * voxel_size_mm))
                ds.SliceThickness = float(voxel_size_mm)
                ds.SpacingBetweenSlices = float(voxel_size_mm)
                
                # Pixel data characteristics
                # For DICOM: Rows = number of samples along Y, Columns = along X. Transpose to (rows, cols)
                pixel_array = slice_data.T.astype(np.int16, copy=False)
                rows, cols = pixel_array.shape
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = 'MONOCHROME2'
                ds.Rows = int(rows)
                ds.Columns = int(cols)
                ds.PixelSpacing = [float(voxel_size_mm), float(voxel_size_mm)]  # [row spacing (mm), col spacing (mm)]
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 1  # Signed
                ds.RescaleIntercept = 0.0
                ds.RescaleSlope = 1.0
                ds.WindowCenter = 40
                ds.WindowWidth = 400
                ds.PixelData = pixel_array.tobytes()
                
                # Create filename
                filename = os.path.join(output_dir, f"CT_Slice_{i+1:04d}.dcm")
                
                # Save the DICOM file
                ds.is_little_endian = True
                ds.is_implicit_VR = False
                ds.save_as(filename)
                
                # Progress
                if progress_callback:
                    progress_callback(i + 1, num_slices)
            except Exception as e:
                print(f"Error saving DICOM file for slice {i+1}: {str(e)}")
                return {'error': f"Error saving DICOM file for slice {i+1}: {str(e)}"}
                
        return {'success': f"Successfully exported {num_slices} DICOM slices to {output_dir}"}

def voxelize_mesh(obj, voxel_size=1.0, padding=1):
    """
    Voxelize a mesh object into a 3D numpy array using a BVH-based column fill.

    This algorithm casts a ray along +Z for each X,Y column, collects all
    intersections, and fills ranges between alternating intersections.
    It dramatically reduces per-voxel mesh queries compared to checking
    each voxel center individually.
    
    Args:
        obj: Blender mesh object to voxelize
        voxel_size: Size of each voxel in Blender units (meters)
        padding: Number of voxels to pad around the object
        
    Returns:
        tuple: (voxel_array, origin, dimensions) where:
            - voxel_array: 3D numpy array (width, height, depth), uint8 {0,1}
            - origin: Vector representing world position of voxel (0,0,0)
            - dimensions: tuple of (width, height, depth) in voxels
    """
    # Build evaluated mesh and BVH for fast intersection
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    bvh = BVHTree.FromObject(obj_eval, depsgraph, epsilon=0.0)

    # Get object's bounding box in world coordinates
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Calculate bounding box min/max
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)
    
    # Add padding
    min_x -= padding * voxel_size
    max_x += padding * voxel_size
    min_y -= padding * voxel_size
    max_y += padding * voxel_size
    min_z -= padding * voxel_size
    max_z += padding * voxel_size
    
    # Calculate voxel grid dimensions
    width = max(1, int(math.ceil((max_x - min_x) / voxel_size)))
    height = max(1, int(math.ceil((max_y - min_y) / voxel_size)))
    depth = max(1, int(math.ceil((max_z - min_z) / voxel_size)))
    
    # Create voxel array (width, height, depth)
    voxel_array = np.zeros((width, height, depth), dtype=np.uint8)
    
    # Origin point (world coordinates of voxel [0,0,0])
    origin = Vector((min_x, min_y, min_z))
    
    print(f"Voxelizing mesh '{obj.name}' into {width}x{height}x{depth} grid (BVH ray casting)...")
    
    # Precompute world coordinates of voxel centers in X and Y
    xs = min_x + (np.arange(width) + 0.5) * voxel_size
    ys = min_y + (np.arange(height) + 0.5) * voxel_size
    
    # Z center reference for index mapping
    z0_center = min_z + 0.5 * voxel_size
    inv_dz = 1.0 / voxel_size
    
    # Ray casting parameters
    ray_dir = Vector((0.0, 0.0, 1.0))
    max_dist = (max_z - min_z) + 4.0 * voxel_size
    eps = 1e-6
    
    # Iterate over columns (X,Y) and fill Z ranges
    total_cols = width * height
    col_count = 0
    
    for ix in range(width):
        xw = float(xs[ix])
        for iy in range(height):
            yw = float(ys[iy])
            # Start ray just below the min Z to ensure first hit is the first surface
            origin_ray = Vector((xw, yw, min_z - 2.0 * voxel_size))
            hits_z = []
            
            # Collect all intersections along +Z
            while True:
                loc, normal, face_index, distance = bvh.ray_cast(origin_ray, ray_dir, max_dist)
                if loc is None:
                    break
                hits_z.append(loc.z)
                # Move origin slightly above the hit to find the next intersection
                origin_ray = Vector((loc.x, loc.y, loc.z + eps))
            
            if hits_z:
                hits_z.sort()
                # Fill every alternating interval [z0, z1)
                for j in range(0, len(hits_z) - 1, 2):
                    z0 = hits_z[j]
                    z1 = hits_z[j + 1]
                    # Determine voxel indices whose centers lie in (z0, z1)
                    start_idx = int(math.ceil((z0 - z0_center) * inv_dz))
                    end_idx = int(math.floor((z1 - z0_center) * inv_dz))
                    if end_idx >= start_idx:
                        # Clamp to array bounds
                        s = max(0, start_idx)
                        e = min(depth - 1, end_idx)
                        if e >= s:
                            voxel_array[ix, iy, s:e + 1] = 1
            
            col_count += 1
            if (col_count % 10000) == 0:
                print(f"  processed columns: {col_count}/{total_cols} ({(col_count/total_cols)*100:.1f}%)")
    
    filled = int(voxel_array.sum())
    print(f"Voxelization complete. Filled voxels: {filled}/{voxel_array.size}")
    
    return voxel_array, origin, (width, height, depth)

# New: voxelize multiple objects into a single HU grid

def voxelize_objects_to_hu(objects, voxel_size=1.0, padding=1, progress_callback=None):
    """
    Voxelize multiple mesh objects into a single HU grid (int16), using per-object HU values.
    The union of all objects is used. Overlapping voxels take the maximum HU.

    Returns: (hu_array, origin, (width, height, depth))
    """
    if not objects:
        raise ValueError("No objects provided for voxelization")

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Global bounding box across all objects
    all_corners = []
    for obj in objects:
        all_corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    min_x = min(corner.x for corner in all_corners)
    max_x = max(corner.x for corner in all_corners)
    min_y = min(corner.y for corner in all_corners)
    max_y = max(corner.y for corner in all_corners)
    min_z = min(corner.z for corner in all_corners)
    max_z = max(corner.z for corner in all_corners)

    # Add padding
    min_x -= padding * voxel_size
    max_x += padding * voxel_size
    min_y -= padding * voxel_size
    max_y += padding * voxel_size
    min_z -= padding * voxel_size
    max_z += padding * voxel_size

    # Grid dimensions
    width = max(1, int(math.ceil((max_x - min_x) / voxel_size)))
    height = max(1, int(math.ceil((max_y - min_y) / voxel_size)))
    depth = max(1, int(math.ceil((max_z - min_z) / voxel_size)))

    origin = Vector((min_x, min_y, min_z))

    print(f"Voxelizing {len(objects)} objects into {width}x{height}x{depth} HU grid...")

    # HU array initialized to AIR
    hu_array = np.full((width, height, depth), int(AIR_DENSITY), dtype=np.int16)

    # Precompute X, Y world coordinates for voxel centers
    xs = min_x + (np.arange(width) + 0.5) * voxel_size
    ys = min_y + (np.arange(height) + 0.5) * voxel_size

    z0_center = min_z + 0.5 * voxel_size
    inv_dz = 1.0 / voxel_size

    ray_dir = Vector((0.0, 0.0, 1.0))
    max_dist = (max_z - min_z) + 4.0 * voxel_size
    eps = 1e-6

    # Prepare BVHs and per-object HU
    obj_data = []
    for obj in objects:
        obj_eval = obj.evaluated_get(depsgraph)
        bvh = BVHTree.FromObject(obj_eval, depsgraph, epsilon=0.0)
        # Get HU from object property, default to DEFAULT_DENSITY
        hu_val = float(getattr(obj, 'dicomator_hu', DEFAULT_DENSITY))
        hu_val = max(MIN_HU_VALUE, min(MAX_HU_VALUE, hu_val))
        obj_data.append((bvh, hu_val, obj.name))

    total_cols = width * height
    col_count = 0

    for ix in range(width):
        xw = float(xs[ix])
        for iy in range(height):
            yw = float(ys[iy])
            # For each object, cast along +Z and fill intervals
            for bvh, hu_val, _name in obj_data:
                origin_ray = Vector((xw, yw, min_z - 2.0 * voxel_size))
                hits_z = []
                while True:
                    loc, normal, face_index, distance = bvh.ray_cast(origin_ray, ray_dir, max_dist)
                    if loc is None:
                        break
                    hits_z.append(loc.z)
                    origin_ray = Vector((loc.x, loc.y, loc.z + eps))
                if hits_z:
                    hits_z.sort()
                    for j in range(0, len(hits_z) - 1, 2):
                        z0 = hits_z[j]
                        z1 = hits_z[j + 1]
                        start_idx = int(math.ceil((z0 - z0_center) * inv_dz))
                        end_idx = int(math.floor((z1 - z0_center) * inv_dz))
                        if end_idx >= start_idx:
                            s = max(0, start_idx)
                            e = min(depth - 1, end_idx)
                            if e >= s:
                                # Assign max HU in overlaps
                                current = hu_array[ix, iy, s:e+1]
                                if current.size:
                                    hu_array[ix, iy, s:e+1] = np.maximum(current, np.int16(hu_val))
            
            col_count += 1
            if progress_callback and (col_count % 5000) == 0:
                progress_callback(col_count, total_cols)
            if (col_count % 20000) == 0:
                print(f"  processed columns: {col_count}/{total_cols} ({(col_count/total_cols)*100:.1f}%)")

    print("Voxelization complete (multi-object HU grid).")
    return hu_array, origin, (width, height, depth)





class MESH_OT_export_dicom(Operator):
    """Export mesh to DICOM files"""
    bl_idname = "mesh.export_dicom"
    bl_label = "Export to DICOM"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'OBJECT')
    
    def execute(self, context):
        if not PYDICOM_AVAILABLE:
            self.report({'ERROR'}, "pydicom library not available")
            return {'CANCELLED'}
        
        # Collect all selected mesh objects
        selected_meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if not selected_meshes:
            self.report({'ERROR'}, "Please select at least one mesh object")
            return {'CANCELLED'}
        
        # Use active object for mode validation but export all selected meshes
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh object")
            return {'CANCELLED'}
        
        # Get properties from the panel
        props = context.scene.dicomator_props
        
        # Check if export directory is specified
        output_dir = props.export_directory
        if not output_dir:
            self.report({'ERROR'}, "Please specify an export directory")
            return {'CANCELLED'}
        
        # Convert Blender relative paths to absolute paths
        if output_dir.startswith('//'):
            # Blender relative path - convert to absolute
            relative_path = output_dir[2:]  # Remove '//' prefix
            
            # Normalize path separators for the current OS
            relative_path = relative_path.replace('/', os.sep).replace('\\', os.sep)
            
            if bpy.data.filepath:
                # If blend file is saved, make path relative to blend file
                blend_dir = os.path.dirname(bpy.data.filepath)
                output_dir = os.path.join(blend_dir, relative_path)
            else:
                # If blend file is not saved, use current working directory
                output_dir = os.path.join(os.getcwd(), relative_path)
        
        # Ensure the path is absolute and normalize it
        output_dir = os.path.abspath(os.path.normpath(output_dir))
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.report({'INFO'}, f"Using export directory: {output_dir}")
        except Exception as e:
            self.report({'ERROR'}, f"Cannot create output directory: {str(e)}")
            return {'CANCELLED'}
        
        # Verify directory is writable
        if not os.access(output_dir, os.W_OK):
            self.report({'ERROR'}, f"Output directory is not writable: {output_dir}")
            return {'CANCELLED'}
        
        try:
            # Convert mm to Blender units (assuming Blender units are in meters)
            voxel_size_mm = _get_float_prop(context.scene.dicomator_props, "grid_resolution", 2.0)
            voxel_size = voxel_size_mm * 0.001  # Convert mm to meters
            
            # Get global object dimensions for safety check across all selected meshes
            bbox_corners = []
            for o in selected_meshes:
                bbox_corners.extend([o.matrix_world @ Vector(corner) for corner in o.bound_box])
            min_x = min(corner.x for corner in bbox_corners)
            max_x = max(corner.x for corner in bbox_corners)
            min_y = min(corner.y for corner in bbox_corners)
            max_y = max(corner.y for corner in bbox_corners)
            min_z = min(corner.z for corner in bbox_corners)
            max_z = max(corner.z for corner in bbox_corners)
            
            # Calculate selection dimensions
            obj_width = max_x - min_x
            obj_height = max_y - min_y
            obj_depth = max_z - min_z
            
            # Estimate voxel grid size (with padding)
            padding = 1
            estimated_width = int(math.ceil((obj_width + 2 * padding * voxel_size) / voxel_size))
            estimated_height = int(math.ceil((obj_height + 2 * padding * voxel_size) / voxel_size))
            estimated_depth = int(math.ceil((obj_depth + 2 * padding * voxel_size) / voxel_size))
            
            # Safety check: prevent extremely large arrays
            max_voxels_per_dimension = 2000  # Reasonable limit
            total_estimated_voxels = estimated_width * estimated_height * estimated_depth
            max_total_voxels = 100_000_000  # 100 million voxels max (~100MB for uint8)
            
            if (estimated_width > max_voxels_per_dimension or 
                estimated_height > max_voxels_per_dimension or 
                estimated_depth > max_voxels_per_dimension or
                total_estimated_voxels > max_total_voxels):
                
                self.report({'ERROR'}, 
                    f"Voxel grid too large: {estimated_width}x{estimated_height}x{estimated_depth} "
                    f"({total_estimated_voxels:,} voxels). "
                    f"Selection size: {obj_width:.3f}x{obj_height:.3f}x{obj_depth:.3f}m. "
                    f"Try increasing grid resolution (current: {voxel_size_mm}mm) or scaling down the objects.")
                return {'CANCELLED'}
            
            # Progress callback function
            def progress_callback(current, total):
                progress = min(1.0, max(0.0, current / max(1, total)))
                context.window_manager.progress_update(progress)
            
            # Start progress indicator
            context.window_manager.progress_begin(0, 1)
            
            self.report({'INFO'}, 
                f"Voxelizing {len(selected_meshes)} mesh(es) with {voxel_size_mm}mm resolution. "
                f"Estimated grid: {estimated_width}x{estimated_height}x{estimated_depth}")
            
            # Build HU grid for all selected meshes
            hu_array, origin, dimensions = voxelize_objects_to_hu(
                selected_meshes,
                voxel_size=voxel_size,
                padding=padding,
                progress_callback=progress_callback,
            )
            
            # Export to DICOM (direct HU values)
            result = export_voxel_grid_to_dicom(
                hu_array,
                voxel_size,
                output_dir,
                origin,
                patient_name=props.patient_name,
                patient_id=props.patient_id,
                patient_sex=props.patient_sex,
                series_description=props.series_description,
                progress_callback=progress_callback,
                direct_hu=True,
                patient_position=props.patient_position,  # NEW: pass UI-selected patient position
            )
            
            # End progress indicator
            context.window_manager.progress_end()
            
            if 'error' in result:
                self.report({'ERROR'}, result['error'])
                return {'CANCELLED'}
            else:
                self.report({'INFO'}, result['success'])
                return {'FINISHED'}
                
        except Exception as e:
            context.window_manager.progress_end()
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}


class DICOMatorProperties(PropertyGroup):
    """Properties for DICOMator add-on"""
    # Switch to annotation style so Blender 4.x registers RNA correctly
    patient_name: bpy.props.StringProperty(
        name="Patient Name",
        description="Name of the patient",
        default="Anonymous"
    )
    patient_id: bpy.props.StringProperty(
        name="MRN",
        description="Medical Record Number (Patient ID)",
        default="12345678"
    )
    patient_sex: bpy.props.EnumProperty(
        name="Patient Sex",
        description="Patient sex",
        items=[
            ('M', 'Male', 'Male'),
            ('F', 'Female', 'Female'),
            ('O', 'Other', 'Other'),
        ],
        default='M'
    )
    # NEW: DICOM PatientPosition (orientation)
    patient_position: bpy.props.EnumProperty(
        name="Patient Position",
        description="DICOM Patient Position (image orientation context)",
        items=[
            ('HFS', 'Head First Supine', 'Head First Supine'),
            ('FFS', 'Feet First Supine', 'Feet First Supine'),
            ('HFP', 'Head First Prone', 'Head First Prone'),
            ('FFP', 'Feet First Prone', 'Feet First Prone'),
            ('HFDR', 'Head First Decubitus Right', 'Head First Decubitus Right'),
            ('HFDL', 'Head First Decubitus Left', 'Head First Decubitus Left'),
            ('FFDR', 'Feet First Decubitus Right', 'Feet First Decubitus Right'),
            ('FFDL', 'Feet First Decubitus Left', 'Feet First Decubitus Left'),
        ],
        default='HFS'
    )
    grid_resolution: bpy.props.FloatProperty(
        name="Grid Resolution (mm)",
        description="Size of each voxel in millimeters",
        default=2.0,
        min=0.1,
        max=10.0,
        step=10,
        precision=2
    )
    export_directory: bpy.props.StringProperty(
        name="Export Directory",
        description="Directory to save DICOM files",
        subtype='DIR_PATH',
        default="C:\\Users\\Public\\DICOM_Export"
    )
    series_description: bpy.props.StringProperty(
        name="Series Description",
        description="Description for the DICOM series",
        default="CT Series from DICOMator"
    )


class VIEW3D_PT_dicomator_panel(Panel):
    """DICOMator panel in the 3D viewport sidebar"""
    bl_label = "DICOMator"
    bl_idname = "VIEW3D_PT_dicomator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    
    def draw(self, context):
        layout = self.layout
        # Keep parent minimal; show message if nothing selected
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="Select a mesh object to export", icon='INFO')
        else:
            layout.label(text="Expand sections below to configure export", icon='TRIA_DOWN')


# NEW: Collapsible subpanels
class VIEW3D_PT_dicomator_selection_info(Panel):
    bl_label = "Selection Info"
    bl_idname = "VIEW3D_PT_dicomator_selection_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.dicomator_props
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="No mesh selected", icon='INFO')
            return

        # Selected meshes and active
        selected_meshes = [o for o in context.selected_objects if o.type == 'MESH']
        obj = context.active_object
        sel_count = len(selected_meshes)
        if sel_count > 1:
            layout.label(text=f"Selected: {sel_count} meshes (Active: {obj.name})", icon='MESH_DATA')
        else:
            layout.label(text=f"Selected: {obj.name}", icon='MESH_DATA')

        # Combined bounds
        bbox_corners = []
        for o in selected_meshes:
            bbox_corners.extend([o.matrix_world @ Vector(corner) for corner in o.bound_box])
        min_x = min(corner.x for corner in bbox_corners)
        max_x = max(corner.x for corner in bbox_corners)
        min_y = min(corner.y for corner in bbox_corners)
        max_y = max(corner.y for corner in bbox_corners)
        min_z = min(corner.z for corner in bbox_corners)
        max_z = max(corner.z for corner in bbox_corners)

        obj_width = max_x - min_x
        obj_height = max_y - min_y
        obj_depth = max_z - min_z

        box = layout.box()
        box.label(text="Selection Info", icon='INFO')
        box.label(text=f"Size: {obj_width:.2f} x {obj_height:.2f} x {obj_depth:.2f} m")

        # Voxel grid estimate
        grid_res_mm = _get_float_prop(props, "grid_resolution", 2.0)
        if grid_res_mm > 0.0:
            voxel_size = grid_res_mm * 0.001
            padding = 1
            est_width = int(math.ceil((obj_width + 2 * padding * voxel_size) / voxel_size))
            est_height = int(math.ceil((obj_height + 2 * padding * voxel_size) / voxel_size))
            est_depth = int(math.ceil((obj_depth + 2 * padding * voxel_size) / voxel_size))
            total_voxels = est_width * est_height * est_depth

            box.label(text=f"Est. Grid: {est_width} x {est_height} x {est_depth}")
            box.label(text=f"Total Voxels: {total_voxels:,}")

            memory_mb = (total_voxels * 2) / (1024 * 1024)
            box.label(text=f"Est. Memory: {memory_mb:.1f} MB")

            if total_voxels > 50_000_000:
                box.label(text="Large grid - may be slow", icon='ERROR')
            elif total_voxels > 100_000_000:
                box.label(text="Grid too large!", icon='CANCEL')


class VIEW3D_PT_dicomator_per_object_hu(Panel):
    bl_label = "Per-Object HU"
    bl_idname = "VIEW3D_PT_dicomator_per_object_hu"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        if not (context.active_object and context.active_object.type == 'MESH'):
            layout.label(text="No mesh selected", icon='INFO')
            return
        selected_meshes = [o for o in context.selected_objects if o.type == 'MESH']
        box_hu = layout.box()
        box_hu.label(text="Per-Object HU", icon='MOD_PHYSICS')
        if selected_meshes:
            for o in selected_meshes:
                row = box_hu.row(align=True)
                row.prop(o, "dicomator_hu", text=f"{o.name} HU")


class VIEW3D_PT_dicomator_patient_info(Panel):
    bl_label = "Patient Information"
    bl_idname = "VIEW3D_PT_dicomator_patient_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.dicomator_props
        box = layout.box()
        box.label(text="Patient Information", icon='USER')
        box.prop(props, "patient_name")
        box.prop(props, "patient_id")
        box.prop(props, "patient_sex")


class VIEW3D_PT_dicomator_orientation(Panel):
    bl_label = "Image Orientation"
    bl_idname = "VIEW3D_PT_dicomator_orientation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.dicomator_props
        box_or = layout.box()
        box_or.label(text="Image Orientation", icon='ORIENTATION_GIMBAL')
        box_or.prop(props, "patient_position", text="Patient Position")


class VIEW3D_PT_dicomator_export_settings(Panel):
    bl_label = "Export Settings"
    bl_idname = "VIEW3D_PT_dicomator_export_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DICOMator"
    bl_parent_id = "VIEW3D_PT_dicomator_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.dicomator_props

        box = layout.box()
        box.label(text="Export Settings", icon='SETTINGS')
        box.prop(props, "grid_resolution")
        box.prop(props, "export_directory")

        # Resolved export path for Blender-relative paths
        export_dir_val = _get_str_prop(props, "export_directory", "")
        if export_dir_val.startswith('//'):
            relative_path = export_dir_val[2:].replace('/', os.sep).replace('\\', os.sep)
            if bpy.data.filepath:
                blend_dir = os.path.dirname(bpy.data.filepath)
                resolved_path = os.path.join(blend_dir, relative_path)
            else:
                resolved_path = os.path.join(os.getcwd(), relative_path)
            resolved_path = os.path.abspath(os.path.normpath(resolved_path))
            box.label(text=f"Resolved: {resolved_path}", icon='FILE_FOLDER')

        box.prop(props, "series_description")

        # Export button and guardrails
        if PYDICOM_AVAILABLE:
            export_dir = _get_str_prop(props, "export_directory", "")
            if export_dir.startswith('//'):
                relative_path = export_dir[2:].replace('/', os.sep).replace('\\', os.sep)
                if bpy.data.filepath:
                    blend_dir = os.path.dirname(bpy.data.filepath)
                    export_dir = os.path.join(blend_dir, relative_path)
                else:
                    export_dir = os.path.join(os.getcwd(), relative_path)
                export_dir = os.path.abspath(os.path.normpath(export_dir))

            if export_dir and export_dir.strip():
                # If a mesh is selected, compute a coarse size check for button gating
                if context.active_object and context.active_object.type == 'MESH':
                    selected_meshes = [o for o in context.selected_objects if o.type == 'MESH']
                    bbox_corners = []
                    for o in selected_meshes:
                        bbox_corners.extend([o.matrix_world @ Vector(corner) for corner in o.bound_box])
                    min_x = min(corner.x for corner in bbox_corners)
                    max_x = max(corner.x for corner in bbox_corners)
                    min_y = min(corner.y for corner in bbox_corners)
                    max_y = max(corner.y for corner in bbox_corners)
                    min_z = min(corner.z for corner in bbox_corners)
                    max_z = max(corner.z for corner in bbox_corners)
                    obj_width = max_x - min_x
                    obj_height = max_y - min_y
                    obj_depth = max_z - min_z

                    grid_res_mm2 = _get_float_prop(props, "grid_resolution", 2.0)
                    if grid_res_mm2 > 0.0:
                        voxel_size = grid_res_mm2 * 0.001
                        est_total = int(math.ceil((obj_width + 2 * voxel_size) / voxel_size)) * \
                                   int(math.ceil((obj_height + 2 * voxel_size) / voxel_size)) * \
                                   int(math.ceil((obj_depth + 2 * voxel_size) / voxel_size))

                        if est_total > 100_000_000:
                            layout.label(text="Grid too large - increase resolution", icon='ERROR')
                        else:
                            layout.operator("mesh.export_dicom", text="Export to DICOM", icon='EXPORT')
                    else:
                        layout.operator("mesh.export_dicom", text="Export to DICOM", icon='EXPORT')
                else:
                    layout.label(text="Select a mesh object to export", icon='INFO')
            else:
                layout.label(text="Please select export directory", icon='INFO')
        else:
            box_err = layout.box()
            box_err.label(text="pydicom library required", icon='ERROR')
            box_err.label(text="Install pydicom to enable DICOM export")


# Registration
classes = (
    DICOMatorProperties,
    MESH_OT_export_dicom,
    VIEW3D_PT_dicomator_panel,
    # NEW: register subpanels
    VIEW3D_PT_dicomator_selection_info,
    VIEW3D_PT_dicomator_per_object_hu,
    VIEW3D_PT_dicomator_patient_info,
    VIEW3D_PT_dicomator_orientation,
    VIEW3D_PT_dicomator_export_settings,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register the property group with the scene
    bpy.types.Scene.dicomator_props = PointerProperty(type=DICOMatorProperties)

    # Register per-object HU property (defaults to 0 HU as requested)
    bpy.types.Object.dicomator_hu = FloatProperty(
        name="HU",
        description="Assigned Hounsfield Units for this mesh",
        default=0.0,
        min=MIN_HU_VALUE,
        max=MAX_HU_VALUE,
        step=10,
        precision=0,
    )

def unregister():
    # Unregister the property group
    del bpy.types.Scene.dicomator_props

    # Unregister per-object HU property
    if hasattr(bpy.types.Object, 'dicomator_hu'):
        del bpy.types.Object.dicomator_hu
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
