# DICOMator Contributor Notes

## Repository Overview
- **Purpose:** Blender 4.2+ add-on that voxelizes mesh objects and exports them as CT-style DICOM series, optionally layering synthetic artifacts.
- **Key modules:**
  - `__init__.py` – Blender registration entry point and exported API.
  - `properties.py` – `bpy.types.PropertyGroup` definitions backing the add-on UI.
  - `panels.py` / `operators.py` – UI panels and operators exposed in the 3D Viewport sidebar.
  - `voxelization.py` – Mesh voxelization helpers that populate voxel grids and estimate selection bounds.
  - `artifacts.py` – Synthetic artifact generators (noise, streaks, rings, partial volume, motion).
  - `dicom_export.py` – Writes voxel grids to DICOM slices via pydicom when available.
  - `constants.py` – Core HU constants and optional imports (pydicom, Dataset helpers).
  - `utils.py` – Lightweight helpers for property access and UI refresh.
  - `download_wheels.py` / `wheels/` – Optional vendor wheels for Blender’s bundled Python.

## Coding Guidelines
- Target **Python 3.11** (matches Blender 4.2 runtime). Use modern type hints and keep `from __future__ import annotations` at the top of new modules.
- Follow **PEP 8** conventions: 4-space indentation, descriptive naming, and module-level docstrings. Keep public helpers exported through `__all__` lists when the surrounding module already uses them.
- Prefer explicit type hints (`-> None`, concrete collection types) and keep docstrings concise but informative. Use f-strings for string interpolation.
- Blender-specific code (`bpy`, `mathutils`) should remain importable without running inside Blender. Avoid executing Blender ops at import time; confine them to functions/operators.
- When updating release metadata, remember to keep `bl_info` in `__init__.py` and `blender_manifest.toml` in sync.
- Avoid committing large binary assets. The `wheels/` directory already contains prebuilt dependencies; keep additions minimal and justify them in commit messages.

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
- Feel free to extend this document with additional conventions as the project evolves.
