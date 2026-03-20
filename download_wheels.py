"""Download optional wheels that match Blender's bundled Python."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _python_tag(version: str) -> str:
    """Return a CPython ABI tag for a version string like ``3.13``."""

    major, minor = version.split('.', 1)
    return f"cp{major}{minor}"


def main() -> None:
    blender_python = os.environ.get("BLENDER_PYTHON_VERSION", "3.13")
    platform_tag = os.environ.get("BLENDER_PLATFORM", "win_amd64")
    implementation = os.environ.get("BLENDER_IMPLEMENTATION", "cp")
    abi_tag = os.environ.get("BLENDER_ABI", _python_tag(blender_python))
    packages = [
        pkg.strip()
        for pkg in os.environ.get("DICOMATOR_WHEEL_PACKAGES", "pydicom==2.3.1").split(',')
        if pkg.strip()
    ]

    wheels_dir = Path("wheels")
    wheels_dir.mkdir(exist_ok=True)

    for package in packages:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "download",
            package,
            "--no-deps",
            "--only-binary=:all:",
            "--python-version",
            blender_python,
            "--platform",
            platform_tag,
            "--implementation",
            implementation,
            "--abi",
            abi_tag,
            "-d",
            str(wheels_dir),
        ]
        print(f"Downloading {package} for Python {blender_python} ({platform_tag}, {abi_tag})")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
