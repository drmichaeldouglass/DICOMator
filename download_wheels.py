import os
import subprocess

def main():
    packages = [
        "scikit-image==0.24.0",
        "pydicom==2.3.1",
        "scipy==1.14.1"
    ]

    wheels_dir = "./wheels"
    os.makedirs(wheels_dir, exist_ok=True)

    for pkg in packages:
        cmd = [
            "python", "-m", "pip", "download",
            pkg,
            "--no-deps",
            "--only-binary=:all:",   # Ensure wheels only
            "--python-version", "3.11",
            "--platform", "win_amd64",
            "--implementation", "cp",
            "--abi", "cp311",
            "-d", wheels_dir
        ]
        print(f"Downloading: {pkg} → {wheels_dir}")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
