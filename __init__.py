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

# Add these imports after the existing imports
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
        progress_callback=None
    ):
        """
        Export the voxel grid as a series of DICOM slices.
        
        Args:
            voxel_grid: 3D numpy array of voxel data (HU values)
            voxel_size: Size of each voxel in world units
            output_dir: Directory to save DICOM files
            bbox_min: Vector representing minimum bounding box coordinates
            patient_name: Name of the patient
            patient_id: Patient ID
            patient_sex: Patient sex (M/F/O)
            series_description: Description for the DICOM series
            progress_callback: Optional callback function(slice_index, total_slices)
            
        Returns:
            Status dictionary
        """
        if not PYDICOM_AVAILABLE:
            return {'error': 'pydicom not available'}
        
        import numpy as np
        import pydicom
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current date and time
        current_datetime = datetime.now()
        date_str = current_datetime.strftime('%Y%m%d')
        time_str = current_datetime.strftime('%H%M%S.%f')
        
        # Get number of slices
        num_slices = voxel_grid.shape[2]
        
        # Convert binary voxel data to HU values
        # 1 = tissue (0 HU), 0 = air (-1000 HU)
        hu_grid = np.where(voxel_grid > 0, DEFAULT_DENSITY, AIR_DENSITY)
        
        # Clip values to valid HU range
        hu_grid = np.clip(hu_grid, MIN_HU_VALUE, MAX_HU_VALUE)
        
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
        
        # Export each slice
        for i in range(num_slices):
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
                    float(bbox_min.x),
                    float(bbox_min.y),
                    float(bbox_min.z + (i * voxel_size))
                ]
                ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                ds.SliceLocation = float(bbox_min.z + (i * voxel_size))
                
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
                filename = os.path.join(output_dir, f"CT_Slice_{i+1:04d}.dcm")
                
                # Save the DICOM file
                ds.is_little_endian = True
                ds.is_implicit_VR = False
                ds.save_as(filename)
                
                print(f"Saved slice {i+1}/{num_slices}")
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, num_slices)
                
            except Exception as e:
                print(f"Error saving DICOM file for slice {i+1}: {str(e)}")
                return {'error': f"Error saving DICOM file for slice {i+1}: {str(e)}"}
                
        return {'success': f"Successfully exported {num_slices} DICOM slices to {output_dir}"}

def voxelize_mesh(obj, voxel_size=1.0, padding=1):
    """
    Voxelize a mesh object into a 3D numpy array.
    
    Args:
        obj: Blender mesh object to voxelize
        voxel_size: Size of each voxel in Blender units
        padding: Number of voxels to pad around the object
        
    Returns:
        tuple: (voxel_array, origin, dimensions) where:
            - voxel_array: 3D numpy array (1 = inside mesh, 0 = outside)
            - origin: Vector representing the world position of voxel (0,0,0)
            - dimensions: tuple of (width, height, depth) in voxels
    """
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
    width = int(math.ceil((max_x - min_x) / voxel_size))
    height = int(math.ceil((max_y - min_y) / voxel_size))
    depth = int(math.ceil((max_z - min_z) / voxel_size))
    
    # Create voxel array
    voxel_array = np.zeros((width, height, depth), dtype=np.uint8)
    
    # Origin point (world coordinates of voxel [0,0,0])
    origin = Vector((min_x, min_y, min_z))
    
    print(f"Voxelizing mesh '{obj.name}' into {width}x{height}x{depth} grid...")
    
    # Fill voxel array
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                # Calculate world position of voxel center
                world_pos = Vector((
                    min_x + (x + 0.5) * voxel_size,
                    min_y + (y + 0.5) * voxel_size,
                    min_z + (z + 0.5) * voxel_size
                ))
                
                # Check if point is inside mesh
                if is_point_inside_mesh(world_pos, obj):
                    voxel_array[x, y, z] = 1
    
    print(f"Voxelization complete. Filled voxels: {np.sum(voxel_array)}/{voxel_array.size}")
    
    return voxel_array, origin, (width, height, depth)





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
            # But first, let's check what units Blender is actually using
            voxel_size_mm = props.grid_resolution
            
            # In Blender, the default unit is typically 1 Blender unit = 1 meter
            # So we need to convert mm to meters: 1mm = 0.001m
            voxel_size = voxel_size_mm * 0.001  # Convert mm to meters
            
            # Get object dimensions for safety check
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            min_x = min(corner.x for corner in bbox_corners)
            max_x = max(corner.x for corner in bbox_corners)
            min_y = min(corner.y for corner in bbox_corners)
            max_y = max(corner.y for corner in bbox_corners)
            min_z = min(corner.z for corner in bbox_corners)
            max_z = max(corner.z for corner in bbox_corners)
            
            # Calculate object dimensions
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
                    f"Object size: {obj_width:.3f}x{obj_height:.3f}x{obj_depth:.3f}m. "
                    f"Try increasing grid resolution (current: {voxel_size_mm}mm) or scaling down the object.")
                return {'CANCELLED'}
            
            # Progress callback function
            def progress_callback(current, total):
                progress = current / total
                context.window_manager.progress_update(progress)
            
            # Start progress indicator
            context.window_manager.progress_begin(0, 1)
            
            self.report({'INFO'}, 
                f"Voxelizing mesh '{obj.name}' with {voxel_size_mm}mm resolution. "
                f"Estimated grid: {estimated_width}x{estimated_height}x{estimated_depth}")
            
            voxel_array, origin, dimensions = voxelize_mesh(
                obj, 
                voxel_size=voxel_size,
                padding=padding
            )
            
            # Export to DICOM
            result = export_voxel_grid_to_dicom(
                voxel_array,
                voxel_size,
                output_dir,
                origin,
                patient_name=props.patient_name,
                patient_id=props.patient_id,
                patient_sex=props.patient_sex,
                series_description=props.series_description,
                progress_callback=progress_callback
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
    
    patient_name: StringProperty(
        name="Patient Name",
        description="Name of the patient",
        default="Anonymous"
    )
    
    patient_id: StringProperty(
        name="MRN",
        description="Medical Record Number (Patient ID)",
        default="12345678"
    )
    
    patient_sex: EnumProperty(
        name="Patient Sex",
        description="Patient sex",
        items=[
            ('M', 'Male', 'Male'),
            ('F', 'Female', 'Female'),
            ('O', 'Other', 'Other'),
        ],
        default='M'
    )
    
    grid_resolution: FloatProperty(
        name="Grid Resolution (mm)",
        description="Size of each voxel in millimeters",
        default=2.0,
        min=0.1,
        max=10.0,
        step=10,
        precision=2
    )
    
    export_directory: StringProperty(
        name="Export Directory",
        description="Directory to save DICOM files",
        subtype='DIR_PATH',
        default="C:\\Users\\Public\\DICOM_Export"
    )
    
    series_description: StringProperty(
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
        props = context.scene.dicomator_props
        
        # Check if there's a selected mesh object
        if context.active_object and context.active_object.type == 'MESH':
            obj = context.active_object
            layout.label(text=f"Selected: {obj.name}", icon='MESH_DATA')
            
            # Show object dimensions
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
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
            box.label(text="Object Info", icon='INFO')
            box.label(text=f"Size: {obj_width:.2f} x {obj_height:.2f} x {obj_depth:.2f} m")
            
            # Calculate and show estimated voxel grid size
            if props.grid_resolution > 0:
                voxel_size = props.grid_resolution * 0.001  # mm to meters
                padding = 1
                est_width = int(math.ceil((obj_width + 2 * padding * voxel_size) / voxel_size))
                est_height = int(math.ceil((obj_height + 2 * padding * voxel_size) / voxel_size))
                est_depth = int(math.ceil((obj_depth + 2 * padding * voxel_size) / voxel_size))
                total_voxels = est_width * est_height * est_depth
                
                box.label(text=f"Est. Grid: {est_width} x {est_height} x {est_depth}")
                box.label(text=f"Total Voxels: {total_voxels:,}")
                
                # Memory estimate (uint8 = 1 byte per voxel)
                memory_mb = total_voxels / (1024 * 1024)
                box.label(text=f"Est. Memory: {memory_mb:.1f} MB")
                
                # Warning for large grids
                if total_voxels > 50_000_000:  # 50 million voxels
                    box.label(text="⚠️ Large grid - may be slow", icon='ERROR')
                elif total_voxels > 100_000_000:  # 100 million voxels
                    box.label(text="❌ Grid too large!", icon='CANCEL')
            
            # Patient Information Section
            box = layout.box()
            box.label(text="Patient Information", icon='USER')
            box.prop(props, "patient_name")
            box.prop(props, "patient_id")
            box.prop(props, "patient_sex")
            
            # Export Settings Section
            box = layout.box()
            box.label(text="Export Settings", icon='SETTINGS')
            box.prop(props, "grid_resolution")
            box.prop(props, "export_directory")
            
            # Show resolved export path if it's a Blender relative path
            if props.export_directory.startswith('//'):
                relative_path = props.export_directory[2:].replace('/', os.sep).replace('\\', os.sep)
                if bpy.data.filepath:
                    blend_dir = os.path.dirname(bpy.data.filepath)
                    resolved_path = os.path.join(blend_dir, relative_path)
                else:
                    resolved_path = os.path.join(os.getcwd(), relative_path)
                resolved_path = os.path.abspath(os.path.normpath(resolved_path))
                box.label(text=f"Resolved: {resolved_path}", icon='FILE_FOLDER')
            
            box.prop(props, "series_description")
            
            # Export Button
            if PYDICOM_AVAILABLE:
                # Resolve export directory path
                export_dir = props.export_directory
                if export_dir.startswith('//'):
                    relative_path = export_dir[2:].replace('/', os.sep).replace('\\', os.sep)
                    if bpy.data.filepath:
                        blend_dir = os.path.dirname(bpy.data.filepath)
                        export_dir = os.path.join(blend_dir, relative_path)
                    else:
                        export_dir = os.path.join(os.getcwd(), relative_path)
                    export_dir = os.path.abspath(os.path.normpath(export_dir))
                
                if export_dir and export_dir.strip():
                    # Check if grid size is reasonable
                    if props.grid_resolution > 0:
                        voxel_size = props.grid_resolution * 0.001
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
                    layout.label(text="Please select export directory", icon='INFO')
            else:
                box = layout.box()
                box.label(text="pydicom library required", icon='ERROR')
                box.label(text="Install pydicom to enable DICOM export")
        else:
            layout.label(text="Select a mesh object to export", icon='INFO')


# Registration
classes = (
    DICOMatorProperties,
    MESH_OT_export_dicom,
    VIEW3D_PT_dicomator_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register the property group with the scene
    bpy.types.Scene.dicomator_props = PointerProperty(type=DICOMatorProperties)

def unregister():
    # Unregister the property group
    del bpy.types.Scene.dicomator_props
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
