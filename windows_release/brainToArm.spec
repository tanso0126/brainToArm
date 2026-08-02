# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the self-contained Windows control center."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


ROOT = Path(SPECPATH).parent
RELEASE = ROOT / "windows_release"

datas = [
    # Top-level bundled modules live in PyInstaller's _internal directory, so
    # their __file__.parent resolves this source release directory here.
    (str(RELEASE / "assets"), "assets"),
    # config.py is bundled under _internal/laptop and loads the exact HOME
    # constants during import, including when only the simulator is opened.
    (str(ROOT / "firmware"), "firmware"),
    (str(ROOT / "simul" / "model_manifest.json"), "simul"),
    (str(ROOT / "simul" / "Robotic+Arm+with+Servo+&+Arduino.zip"), "simul"),
    (str(ROOT / "simul" / "Robotic+Arm+with+Servo+&+Arduino.3mf"), "simul"),
    (str(ROOT / "simul" / "models"), "simul/models"),
]
binaries = []
hiddenimports = [
    "config",
    "cognitive_load",
    "errp",
    "floor_grasp",
    "polyg_hid",
    "realtime_visual_servo",
    "windows_camera",
]
hiddenimports += collect_submodules("simul")

for package in ("ultralytics", "mujoco", "trimesh", "webview"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

analysis = Analysis(
    [str(RELEASE / "control_center.py")],
    pathex=[str(ROOT), str(ROOT / "laptop"), str(RELEASE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "numpy.tests"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="brainToArm",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="brainToArm",
)
