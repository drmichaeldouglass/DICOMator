# DICOMator Contributor Notes

## Repository Overview
- **Purpose:** Blender 5.1+ add-on that voxelizes mesh objects and exports them as CT-style DICOM series, optionally layering synthetic artifacts, and can generate digitally reconstructed radiographs (DRRs) and synthetic radiotherapy data (RT-DOSE and RT-STRUCT).
- **Key modules:**
  - `__init__.py` – Blender registration entry point and exported API.
  - `properties.py` – `bpy.types.PropertyGroup` definitions backing the add-on UI.
  - `panels.py` / `operators.py` – UI panels and operators exposed in the 3D Viewport sidebar.
  - `voxelization.py` – Mesh voxelization helpers that populate voxel grids and estimate selection bounds.
  - `artifacts.py` – Synthetic artifact generators (noise, streaks, rings, partial volume, motion). Artifacts should be grounded by physical models where possible. 
  - `dicom_export.py` – Writes voxel grids to DICOM slices via pydicom when available.
  - `constants.py` – Core HU constants and optional imports (pydicom, Dataset helpers), plus the Basic/Intermediate/Advanced UI mode table and the helpers that resolve which settings a mode applies.
  - `utils.py` – Lightweight helpers for property access and UI refresh.
  - `drr.py` – Digitally Reconstructed Radiograph (DRR) generator; performs ray-casting through the voxel HU grid to simulate planar X-ray projections.
  - `rtdose_export.py` – Exports a synthetic 3D dose distribution as an RT-DOSE DICOM object via pydicom.
  - `rtstruct_export.py` – Exports mesh-derived contours as an RT-STRUCT DICOM object (Region of Interest sequences) via pydicom.
  - `download_wheels.py` / `wheels/` – Unmodified PyPI wheels declared in the extension manifest and installed by Blender.

## Coding Guidelines
- Target **Python 3.13** (matches the Blender 5.1 runtime).
- Follow **PEP 8** conventions: 4-space indentation, descriptive naming, and module-level docstrings. Keep public helpers exported through `__all__` lists when the surrounding module already uses them.
- Prefer explicit type hints (`-> None`, concrete collection types) and keep docstrings concise but informative. Use f-strings for string interpolation.
- Blender-specific code (`bpy`, `mathutils`) should remain importable without running inside Blender. Avoid executing Blender ops at import time; confine them to functions/operators.
- `blender_manifest.toml` is the authoritative release metadata. Extension add-ons must not reintroduce legacy `bl_info` metadata.
- Keep the add-on licensed as `GPL-3.0-or-later`, use DICOMator-specific Blender identifiers, and declare any new file, network, clipboard, camera, or microphone access in the manifest.
- Avoid committing large binary assets. The `wheels/` directory already contains prebuilt dependencies; keep additions minimal and justify them in commit messages.
- `[build] paths_exclude_pattern` in the manifest **replaces** Blender's built-in default exclude list, so every new development-only path (including hidden tool caches) has to be named there or it ships to users. `tests/test_extension_package.py` fails when a repository-root entry is neither packaged nor excluded.
- Peak-memory figures in `constants.estimate_peak_memory_bytes` are measured with `tracemalloc`, not guessed; re-measure when a pipeline stage changes how many volume-sized arrays it holds.
- Use python modules packaged with Blender where possible and avoid using third party modules which need to be packaged using wheels.
- Ensure the code is written and structured in a way that is easily understandable by a medical physicist.

## Testing & Validation
- A headless pytest suite lives in `tests/` (stub `bpy`/`bmesh`/`mathutils` modules are installed by `tests/conftest.py`, so no Blender is required). Because those stubs make `bpy.props.*` return `None`, `tests/test_blender_wiring.py` checks the Blender-facing surface (property names used in panels/operators, panel parents, register/unregister symmetry) by reading the source instead — keep it in mind when adding a property or panel. **Before committing changes, run (from the repository root):**
  ```bash
  python -m compileall .
  ruff check .
  pytest -q
  blender --command extension validate
  blender --command extension build --output-filepath dicomator.zip
  ```
  Dependencies for the test run: `pip install numpy pydicom pytest ruff`.
- CI (`.github/workflows/ci.yml`) runs the same checks, validates and builds the extension package, and performs real Blender registration smoke tests.
- For features that touch Blender interaction, perform a quick manual smoke test inside Blender if possible (not enforced here, but recommended).

## Documentation & Communication
- Update `README.md` when you add or remove user-facing features or major workflow changes.
- Keep commit messages and PR descriptions focused on functionality and testing (mention manual Blender verification when applicable).
- 
