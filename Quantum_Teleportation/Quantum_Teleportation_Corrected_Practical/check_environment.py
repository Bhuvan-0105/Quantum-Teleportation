"""Check that the required practical packages are installed."""
import sys

print("Python:", sys.version)

packages = ["numpy", "matplotlib", "pyzx", "discopy"]
all_ok = True
for name in packages:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "version not exposed")
        print(f"PASS  {name}: {version}")
    except Exception as exc:
        all_ok = False
        print(f"FAIL  {name}: {exc}")

if not all_ok:
    raise SystemExit("Environment incomplete. Install requirements.txt first.")

print("Environment check passed.")