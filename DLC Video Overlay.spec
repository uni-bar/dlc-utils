# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['dlc_video_overlay.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('README.md', '.'),
        ('retrain_dlc_from_manual_labels.py', '.'),
        ('configs/head_only_config.yaml', 'configs'),
    ],
    hiddenimports=['PyQt5', 'cv2', 'pandas', 'numpy', 'pyarrow', 'fastparquet', 'filterpy', 'filterpy.kalman'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'mpl_toolkits'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DLC Video Overlay',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DLC Video Overlay',
)
app = BUNDLE(
    coll,
    name='DLC Video Overlay.app',
    icon=None,
    bundle_identifier='com.preytouch.dlcvideooverlay',
)
