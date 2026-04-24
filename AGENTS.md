# DICOMator Contributor Notes

## Repository Overview
- **Purpose:** Blender 5.1.1+ add-on that voxelizes mesh objects and exports them as CT-style DICOM series, optionally layering synthetic artifacts, and can generate digitally reconstructed radiographs (DRRs) and synthetic radiotherapy data (RT-DOSE and RT-STRUCT).
- **Key modules:**
  - `__init__.py` – Blender registration entry point and exported API.
  - `properties.py` – `bpy.types.PropertyGroup` definitions backing the add-on UI.
  - `panels.py` / `operators.py` – UI panels and operators exposed in the 3D Viewport sidebar.
  - `voxelization.py` – Mesh voxelization helpers that populate voxel grids and estimate selection bounds.
  - `artifacts.py` – Synthetic artifact generators (noise, streaks, rings, partial volume, motion). Artifacts should be grounded by physical models where possible. 
  - `dicom_export.py` – Writes voxel grids to DICOM slices via pydicom when available.
  - `constants.py` – Core HU constants and optional imports (pydicom, Dataset helpers).
  - `utils.py` – Lightweight helpers for property access and UI refresh.  - `drr.py` – Digitally Reconstructed Radiograph (DRR) generator; performs ray-casting through the voxel HU grid to simulate planar X-ray projections.
  - `rtdose_export.py` – Exports a synthetic 3D dose distribution as an RT-DOSE DICOM object via pydicom.
  - `rtstruct_export.py` – Exports mesh-derived contours as an RT-STRUCT DICOM object (Region of Interest sequences) via pydicom.  - `download_wheels.py` / `wheels/` – Optional vendor wheels for Blender’s bundled Python.

## Coding Guidelines
- Target **Python 3.13** (matches Blender 5.1.1 runtime).
- Follow **PEP 8** conventions: 4-space indentation, descriptive naming, and module-level docstrings. Keep public helpers exported through `__all__` lists when the surrounding module already uses them.
- Prefer explicit type hints (`-> None`, concrete collection types) and keep docstrings concise but informative. Use f-strings for string interpolation.
- Blender-specific code (`bpy`, `mathutils`) should remain importable without running inside Blender. Avoid executing Blender ops at import time; confine them to functions/operators.
- When updating release metadata, remember to keep `bl_info` in `__init__.py` and `blender_manifest.toml` in sync.
- Avoid committing large binary assets. The `wheels/` directory already contains prebuilt dependencies; keep additions minimal and justify them in commit messages.
- Use python modules packaged with Blender where possible and avoid using third party modules which need to be packaged using wheels.
- Ensure the code is written and structured in a way that is easily understandable by a medical physicist.

## Testing & Validation
- The repository does not ship automated unit tests. **Before committing changes, run:**
  ```bash
  python -m compileall DICOMator
  ```
  This catches syntax errors without requiring Blender.
- For features that touch Blender interaction, perform a quick manual smoke test inside Blender if possible (not enforced here, but recommended).

## Documentation & Communication
- Update `README.md` when you add or remove user-facing features or major workflow changes.
- Keep commit messages and PR descriptions focused on functionality and testing (mention manual Blender verification when applicable).
- 
