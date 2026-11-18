"""Render the README artifact galleries.

A numerical thorax phantom (NumPy only, no Blender) is passed through the
add-on's own artifact generators and DICOM slice writer; the written slices
are read back with pydicom and tiled into a labelled PNG.

Run from the repository root (needs numpy, pydicom and Pillow):

    python docs/make_artifact_figures.py
"""
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pydicom

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "assets"
sys.path.insert(0, str(REPO / "tests"))
from conftest import load_module  # noqa: E402
import mathutils  # stub

art = load_module("artifacts")
dx = load_module("dicom_export")
C = load_module("constants")

W, H, D = 240, 180, 9
VOX = (0.0015, 0.0015, 0.003)

def phantom(modality):
    v = lambda key: C.get_material_intensity(key, modality)
    bg = -1000 if modality == "CT" else 0
    x = (np.arange(W) - W / 2 + 0.5)[:, None]
    y = (np.arange(H) - H / 2 + 0.5)[None, :]
    g = np.full((W, H), bg, dtype=np.int16)
    ell = lambda cx, cy, a, b: ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0
    g[ell(0, 0, 105, 75)] = v("FAT")
    g[ell(0, 0, 97, 67)] = v("MUSCLE")
    g[ell(-48, -8, 34, 44)] = v("LUNG")
    g[ell(48, -8, 34, 44)] = v("LUNG")
    g[ell(0, -6, 20, 24)] = v("BLOOD_ACUTE")       # heart
    g[ell(0, 40, 15, 13)] = v("CORTICAL_BONE")     # vertebral body
    g[ell(0, 40, 11, 9)] = v("TRABECULAR_BONE")
    g[ell(0, 58, 5, 5)] = v("CSF_WATER")           # spinal canal
    if modality == "CT":
        g[ell(-22, 48, 4, 4)] = v("TITANIUM")      # pedicle screws
        g[ell(22, 48, 4, 4)] = v("TITANIUM")
    return np.repeat(g[:, :, None], D, axis=2)

def export_mid_slice(vol, modality):
    out = Path(tempfile.mkdtemp())
    res = dx.export_voxel_grid_to_dicom(
        vol, VOX, str(out), mathutils.Vector((0, 0, 0)), direct_hu=True,
        dicom_modality="CT" if modality == "CT" else "MR",
        mr_weighting=None if modality == "CT" else modality)
    assert "error" not in res, res
    files = sorted(out.glob("*.dcm"))
    ds = pydicom.dcmread(files[len(files) // 2])
    px = ds.pixel_array.astype(np.float64)
    if modality == "CT":
        px = px * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    shutil.rmtree(out)
    return px

def window(px, centre, width):
    lo = centre - width / 2
    return (np.clip((px - lo) / width, 0, 1) * 255).astype(np.uint8)

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)

def montage(tiles, cols, path):
    pad, cap = 8, 26
    rows = -(-len(tiles) // cols)
    sheet = Image.new("L", (cols * (W + pad) + pad, rows * (H + cap + pad) + pad), 24)
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        ox, oy = pad + c * (W + pad), pad + r * (H + cap + pad)
        sheet.paste(Image.fromarray(img), (ox, oy))
        draw.text((ox + 2, oy + H + 4), label, fill=235, font=FONT)
    sheet.save(path, optimize=True)
    print(path, sheet.size, Path(path).stat().st_size)

rng = lambda k: np.random.default_rng(k)
ct = phantom("CT")
ct_cases = [
    ("No artifacts", ct),
    ("Gaussian noise (20 HU)", art.add_gaussian_noise(ct, 20.0, rng=rng(1))),
    ("Quantum noise (scale 150)", art.add_poisson_noise(ct, 150.0, rng=rng(2))),
    ("Partial volume (kernel 5)", art.apply_partial_volume_effect(ct, kernel_size=5)),
    ("Metal streaks", art.add_metal_artifacts(ct, intensity=400.0, density_threshold=2000.0, num_streaks=10, falloff=6.0, rng=rng(3))),
    ("Rings (80 HU)", art.add_ring_artifacts(ct, ring_intensity=80.0, ring_radius=0.5, thickness=0.02, jitter=0.02, rng=rng(4))),
    ("Motion (length 9, X)", art.add_motion_artifact(ct, blur_size=9, severity=0.5, axis=0, rng=rng(5))),
]
montage([(l, window(export_mid_slice(v, "CT"), 40, 400)) for l, v in ct_cases], 4, str(OUT) + "/artifacts-ct.png")

mr = phantom("MRI_T1")
vs = (1.5, 1.5, 3.0)
nn = lambda v: np.maximum(v, 0)
mr_cases = [
    ("No artifacts (T1)", mr),
    ("Rician noise (σ 20)", nn(art.add_rician_noise(mr, sigma=20.0, rng=rng(1)))),
    ("Bias field (0.5)", nn(art.add_bias_field_shading(mr, strength=0.5, scale=0.3, rng=rng(2), voxel_size=vs))),
    ("Geometric distortion", nn(art.add_mri_geometric_distortion(mr, gradient_strength=0.1, b0_strength=5.0, b0_scale=0.35, readout_axis=1, rng=rng(3), voxel_size=vs))),
    ("Gibbs ringing", nn(art.add_gibbs_ringing(mr, strength=1.0, truncation=0.3))),
    ("Motion (length 9, Y)", nn(art.add_motion_artifact(mr, blur_size=9, severity=0.5, axis=1, rng=rng(5), fill_value=0.0))),
]
montage([(l, window(export_mid_slice(v, "MRI_T1"), 128, 256)) for l, v in mr_cases], 3, str(OUT) + "/artifacts-mr.png")
