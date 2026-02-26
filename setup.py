# DLC Video Overlay - Easy Mac Installation
#
# This creates a Mac app that you can double-click to run without Python

from setuptools import setup

APP = ['dlc_video_overlay.py']
DATA_FILES = ['README.md']
OPTIONS = {
    'argv_emulation': True,
    'iconfile': None,
    'plist': {
        'CFBundleName': 'DLC Video Overlay',
        'CFBundleDisplayName': 'DLC Video Overlay',
        'CFBundleGetInfoString': "DLC Video Overlay Tool",
        'CFBundleIdentifier': "com.preytouch.dlcvideooverlay",
        'CFBundleVersion': "1.0.0",
        'CFBundleShortVersionString': "1.0.0",
        'NSHumanReadableCopyright': "PreyTouch Lab",
        'NSHighResolutionCapable': True,
    },
    'packages': ['PyQt5', 'cv2', 'pandas', 'numpy'],
    'includes': ['pyarrow', 'fastparquet'],
}

setup(
    name='DLC Video Overlay',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
