import os
import sys
from cx_Freeze import setup, Executable

__version__ = '1.0'

base = None
#base = 'Win32GUI'

setup(
description='WinkDetector',
version=__version__,
executables=[Executable(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'winkdetector.py'), base=base)],
options = {'build_exe': {
'packages': [],
'excludes': [],
'include_msvcr': True,
'includes': ["matplotlib.pyplot", "os", "dlib", "cv2", "playsound"],
'build_exe': f"program",
}},
)
