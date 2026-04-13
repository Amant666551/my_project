# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

PROJECT_DIR = Path.cwd()
SPEC_DIR = PROJECT_DIR / "desktop"
WEB_DIR = PROJECT_DIR / "web"
ASSETS_DIR = SPEC_DIR / "assets"
ICON_FILE = ASSETS_DIR / "app.ico"
BUILD_MODE = os.getenv("DESKTOP_BUILD_MODE", "onedir").strip().lower()
if BUILD_MODE not in {"onedir", "onefile"}:
    BUILD_MODE = "onedir"

datas = []
if WEB_DIR.exists():
    datas.append((str(WEB_DIR), "web"))
if ASSETS_DIR.exists():
    datas.append((str(ASSETS_DIR), "desktop/assets"))
datas += collect_data_files("webview")
datas += collect_data_files("audiolab")
datas += collect_data_files("pyrnnoise")
datas += collect_data_files("speechbrain", include_py_files=True)

binaries = []
binaries += collect_dynamic_libs("pyrnnoise", destdir="pyrnnoise")

hiddenimports = [
    "orchestrator",
    "main",
]
hiddenimports += collect_submodules("webview")
hiddenimports += collect_submodules("audiolab")
hiddenimports += collect_submodules("pyrnnoise")
hiddenimports += collect_submodules("speechbrain")


a = Analysis(
    [str(SPEC_DIR / "desktop_app.py")],
    pathex=[str(PROJECT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if BUILD_MODE == "onefile":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="SpeechTranslator",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(ICON_FILE) if ICON_FILE.exists() else None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="SpeechTranslator",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(ICON_FILE) if ICON_FILE.exists() else None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="SpeechTranslator",
    )
