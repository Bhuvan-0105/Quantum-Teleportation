"""Open generated result PNGs with the operating system's default image viewer.

This intentionally does NOT use Matplotlib's GUI backends, so it works even when
Tk/Tcl is not installed correctly on Windows.
"""
from pathlib import Path
import os
import platform
import subprocess

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

FILES = [
    "01_discopy_coherent_teleportation_circuit.png",
    "02_pyzx_zx_before_simplification.png",
    "03_pyzx_zx_after_full_reduce.png",
    "04_discopy_zx_imported_from_pyzx.png",
    "05_branch_verification.png",
    "06_random_state_fidelity.png",
]

existing = [RESULTS / name for name in FILES if (RESULTS / name).exists()]
missing = [name for name in FILES if not (RESULTS / name).exists()]

for name in missing:
    print(f"Missing: {RESULTS / name}")

if not existing:
    raise SystemExit(
        "No result images found. First run:\n"
        "  python quantum_teleportation_pyzx_discopy.py"
    )

system = platform.system()
print(f"Opening {len(existing)} result image(s) with the default image viewer...")

for path in existing:
    try:
        if system == "Windows":
            # Force each image into an independent Windows Photo Viewer instance
            subprocess.Popen([
                "rundll32.exe",
                "shimgvw.dll,ImageView_Fullscreen",
                str(path)
            ]) 
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        print(f"Could not open {path.name}: {exc}")

print(f"Results folder: {RESULTS}")