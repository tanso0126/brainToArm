"""Hardware-free verification run by setup_windows.ps1."""

from pathlib import Path
import hashlib
import importlib
import sys


RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parent
ASSET = RELEASE / "assets" / "FastSAM-s.pt"
EXPECTED_MODEL_SHA256 = (
    "c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844")


def main():
    required = ("numpy", "cv2", "serial", "ultralytics")
    for name in required:
        importlib.import_module(name)
        print(f"[verify] import {name}: OK")
    if not ASSET.exists() or ASSET.stat().st_size < 20_000_000:
        raise RuntimeError(
            f"bundled FastSAM model is missing or truncated: {ASSET}")
    digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    if digest != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"bundled FastSAM model checksum mismatch: {digest}")
    sys.path.insert(0, str(ROOT / "laptop"))
    sys.path.insert(0, str(RELEASE))
    import config
    config.PLANAR_VISION_MODEL = str(ASSET)
    from vision_segment import FastSAMDetector
    FastSAMDetector()
    for name in ("windows_app", "windows_camera", "windows_support"):
        importlib.import_module(name)
    print(f"[verify] bundled FastSAM model: OK ({digest[:12]}...)")
    print("[verify] Windows launcher modules: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
