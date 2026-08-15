"""Detector geometry emitted alongside a DRR projection.

``Camera.view_frame`` returns corners in camera-local space, while the rays and
``ImagePositionPatient`` are built through ``matrix_world`` — which also carries
the camera object's scale. The reported ``PixelSpacing`` therefore has to be
measured in world space too, otherwise a scaled camera produces a projection
whose declared pixel size disagrees with the geometry it was cast through.

Blender's ``mathutils`` is stubbed in this test suite, so a minimal vector and
matrix implementation is supplied locally for the few operations ``drr`` needs.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import load_module

drr = load_module("drr")


class Vec3:
    """Minimal ``mathutils.Vector`` stand-in supporting the operations used."""

    __slots__ = ("x", "y", "z")

    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self.x, self.y, self.z = (float(value) for value in seq)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __len__(self):
        return 3

    def __getitem__(self, index):
        # NumPy discovers sequences through __getitem__, not __iter__.
        return (self.x, self.y, self.z)[index]

    def __add__(self, other):
        return Vec3((self.x + other.x, self.y + other.y, self.z + other.z))

    def __sub__(self, other):
        return Vec3((self.x - other.x, self.y - other.y, self.z - other.z))

    def __mul__(self, factor):
        return Vec3((self.x * factor, self.y * factor, self.z * factor))

    __rmul__ = __mul__

    def __truediv__(self, divisor):
        return self * (1.0 / float(divisor))

    @property
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self):
        length = self.length
        return self / length if length else Vec3()


class Mat3:
    """Row-major 3x3 rotation/scale matrix."""

    def __init__(self, rows):
        self.rows = [tuple(float(value) for value in row) for row in rows]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self.rows[index]

    def __matmul__(self, vector):
        return Vec3(
            tuple(
                row[0] * vector.x + row[1] * vector.y + row[2] * vector.z
                for row in self.rows
            )
        )


class Mat4:
    """Row-major 4x4 transform built from a uniform scale and a translation."""

    def __init__(self, scale, translation):
        self.scale = float(scale)
        self.translation = Vec3(translation)

    def to_3x3(self):
        s = self.scale
        return Mat3([(s, 0.0, 0.0), (0.0, s, 0.0), (0.0, 0.0, s)])

    def __matmul__(self, vector):
        return self.to_3x3() @ vector + self.translation


class _CameraData:
    type = "ORTHO"

    def view_frame(self, scene=None):
        # Blender reports the frame at one unit in front of the camera; a
        # 2 x 2 unit orthographic frame keeps the arithmetic obvious.
        return [(1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0)]


class _Camera:
    type = "CAMERA"

    def __init__(self, scale=1.0, height=10.0):
        self.data = _CameraData()
        # Looking straight down -Z from above the volume.
        self.matrix_world = Mat4(scale, (0.0, 0.0, height))


class _Scene:
    class render:
        resolution_x = 4
        resolution_y = 2
        resolution_percentage = 100


DETECTOR_WIDTH = 4
DETECTOR_HEIGHT = 2
LOCAL_FRAME_WIDTH_M = 2.0
LOCAL_FRAME_HEIGHT_M = 2.0
# Voxels large enough that the 2 x 2 x 2 volume spans 1.2 m and is therefore
# crossed by the inner detector pixels of the 2 m wide camera frame.
VOXEL_SIZE_M = (0.6, 0.6, 0.6)


@pytest.fixture
def patched_vector(monkeypatch):
    monkeypatch.setattr(drr, "Vector", Vec3)


#: The grid spans z in [-0.2, 1.0] under VOXEL_SIZE_M.
GRID_ORIGIN = (-0.6, -0.6, -0.2)
GRID_TOP_Z = GRID_ORIGIN[2] + 2 * VOXEL_SIZE_M[2]


def _project(camera_scale: float, camera_height: float = 10.0):
    volume = np.full((2, 2, 2), 500.0, dtype=np.int16)
    origin = Vec3(GRID_ORIGIN)
    return drr.generate_drr_from_hu_volume(
        volume,
        VOXEL_SIZE_M,
        origin,
        _Scene(),
        _Camera(camera_scale, camera_height),
    )


def test_detector_size_follows_render_resolution(patched_vector):
    _image, metadata = _project(1.0)
    assert metadata["detector_size"] == (DETECTOR_WIDTH, DETECTOR_HEIGHT)
    assert metadata["spatial_geometry_valid"] is True


def test_unscaled_camera_reports_local_frame_spacing(patched_vector):
    image, metadata = _project(1.0)
    assert image.shape == (DETECTOR_HEIGHT, DETECTOR_WIDTH)
    # Rays must actually cross the volume, otherwise the geometry below is
    # being checked against an empty projection.
    assert int(image.max()) > 0

    row_mm, column_mm = metadata["pixel_spacing_mm"]
    assert row_mm == pytest.approx(LOCAL_FRAME_HEIGHT_M / DETECTOR_HEIGHT * 1000.0)
    assert column_mm == pytest.approx(LOCAL_FRAME_WIDTH_M / DETECTOR_WIDTH * 1000.0)


def test_pixel_spacing_tracks_camera_object_scale(patched_vector):
    """A camera scaled by 2 spans twice the world extent per detector pixel."""

    _image, unscaled = _project(1.0)
    _image, scaled = _project(2.0)

    assert scaled["pixel_spacing_mm"][0] == pytest.approx(
        2.0 * unscaled["pixel_spacing_mm"][0]
    )
    assert scaled["pixel_spacing_mm"][1] == pytest.approx(
        2.0 * unscaled["pixel_spacing_mm"][1]
    )


def test_close_orthographic_camera_still_integrates_the_whole_grid(patched_vector):
    """Parallel rays must not be clipped by the view-frame plane.

    Blender's view frame sits one unit in front of the camera, so a camera
    less than a unit away launches its rays from below the grid: entry
    distances clamp to zero and the projection comes out blank. Because
    orthographic rays carry no perspective, moving the camera along its own
    axis must not change the image at all.
    """

    # Frame plane at 0.5 - 1.0 = -0.5, i.e. below the grid's z range entirely.
    camera_height = 0.5
    assert camera_height - 1.0 < GRID_ORIGIN[2]

    near_image, _metadata = _project(1.0, camera_height=camera_height)
    far_image, _metadata = _project(1.0, camera_height=10.0)

    assert int(near_image.max()) > 0
    np.testing.assert_array_equal(near_image, far_image)


def test_orthographic_projection_is_invariant_to_camera_distance(patched_vector):
    reference, _metadata = _project(1.0, camera_height=10.0)
    for camera_height in (0.5, 1.0, 1.5, 3.0, 50.0):
        image, _metadata = _project(1.0, camera_height=camera_height)
        np.testing.assert_array_equal(image, reference)


def test_pixel_spacing_matches_first_pixel_offset_from_the_frame_corner(patched_vector):
    """PixelSpacing and ImagePositionPatient must describe the same detector.

    The first pixel centre sits half a pixel in from the top-left corner along
    both detector axes, so the corner-to-centre offset is a direct cross-check
    that the two tags were derived from the same world-space frame.
    """

    for camera_scale in (1.0, 2.0, 0.5):
        _image, metadata = _project(camera_scale)
        row_mm, column_mm = metadata["pixel_spacing_mm"]
        position = np.array(metadata["image_position_patient"], dtype=np.float64)
        orientation = np.array(metadata["image_orientation_patient"], dtype=np.float64)
        row_axis, column_axis = orientation[:3], orientation[3:]

        top_left_corner_mm = np.array(
            [
                camera_scale * -LOCAL_FRAME_WIDTH_M / 2.0 * 1000.0,
                camera_scale * LOCAL_FRAME_HEIGHT_M / 2.0 * 1000.0,
                (10.0 - camera_scale) * 1000.0,
            ]
        )
        expected = top_left_corner_mm + 0.5 * column_mm * row_axis + 0.5 * row_mm * column_axis
        np.testing.assert_allclose(position, expected, atol=1e-6)
