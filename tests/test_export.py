import sys
import types
from pathlib import Path

# Stub Blender-related modules
bpy = types.ModuleType('bpy')
bpy.types = types.ModuleType('bpy.types')
bpy.types.Operator = type('Operator', (object,), {})
bpy.types.Panel = type('Panel', (object,), {})
bpy.types.PropertyGroup = type('PropertyGroup', (object,), {})
bpy.utils = types.SimpleNamespace(register_class=lambda cls: None,
                                  unregister_class=lambda cls: None)

sys.modules.setdefault('bpy', bpy)
sys.modules.setdefault('bpy.types', bpy.types)
sys.modules.setdefault('bmesh', types.ModuleType('bmesh'))

props = types.ModuleType('bpy.props')
for name in ['FloatProperty', 'BoolProperty', 'IntProperty',
             'PointerProperty', 'StringProperty', 'EnumProperty']:
    setattr(props, name, lambda *a, **k: None)
sys.modules.setdefault('bpy.props', props)

mathutils = types.ModuleType('mathutils')
class Vector:
    def __init__(self, seq=(0, 0, 0)):
        self.x, self.y, self.z = seq
mathutils.Vector = Vector
sys.modules.setdefault('mathutils', mathutils)

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = types.ModuleType('numpy')
    class ndarray:
        def __init__(self, data):
            self.data = data
        @property
        def shape(self):
            def _shape(d):
                if isinstance(d, list):
                    return (len(d),) + _shape(d[0])
                return ()
            return _shape(self.data)
        def __gt__(self, value):
            def _gt(d):
                if isinstance(d, list):
                    return [_gt(x) for x in d]
                return 1 if d > value else 0
            return ndarray(_gt(self.data))
        def __getitem__(self, idx):
            if not isinstance(idx, tuple):
                return self.data[idx]
            d = self.data
            for k in idx:
                if isinstance(k, slice):
                    start, stop, step = k.indices(len(d))
                    d = [d[i] for i in range(start, stop, step)]
                else:
                    d = d[k]
            return ndarray(d) if isinstance(d, list) else d
        def astype(self, _dtype):
            flat = []
            def _flatten(v):
                if isinstance(v, list):
                    for x in v:
                        _flatten(x)
                else:
                    flat.append(int(v))
            _flatten(self.data)
            return _Int16Bytes(flat)
    class _Int16Bytes:
        def __init__(self, data):
            self.data = data
        def tobytes(self):
            import struct
            return b''.join(struct.pack('<h', v) for v in self.data)
    def array(obj):
        return ndarray(obj)
    def where(cond, a, b):
        cond = cond.data if isinstance(cond, ndarray) else cond
        result = []
        for x in range(len(cond)):
            plane = []
            for y in range(len(cond[0])):
                row = []
                for z in range(len(cond[0][0])):
                    row.append(a if cond[x][y][z] else b)
                plane.append(row)
            result.append(plane)
        return ndarray(result)
    def clip(arr, a_min, a_max):
        arr = arr.data if isinstance(arr, ndarray) else arr
        result = []
        for x in range(len(arr)):
            plane = []
            for y in range(len(arr[0])):
                row = []
                for z in range(len(arr[0][0])):
                    v = arr[x][y][z]
                    if v < a_min:
                        v = a_min
                    elif v > a_max:
                        v = a_max
                    row.append(v)
                plane.append(row)
            result.append(plane)
        return ndarray(result)
    int16 = 'int16'
    np.ndarray = ndarray
    np.array = array
    np.where = where
    np.clip = clip
    np.int16 = int16
    sys.modules['numpy'] = np

import pydicom
import importlib
sys.path.append(str(Path(__file__).resolve().parents[2]))
import DICOMator as dicomator


def test_export_creates_files(tmp_path):
    voxel = np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ])
    out_dir = tmp_path / "dicom"
    bbox = Vector((0, 0, 0))
    result = dicomator.export_voxel_grid_to_dicom(
        voxel, 1.0, str(out_dir), bbox, patient_name="Test"
    )
    assert "success" in result
    files = sorted(out_dir.glob("*.dcm"))
    assert len(files) == voxel.shape[2]
    ds = pydicom.dcmread(str(files[0]))
    assert ds.PatientName == "Test"
