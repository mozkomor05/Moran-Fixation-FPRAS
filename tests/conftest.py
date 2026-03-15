"""Add build-python/ to sys.path so 'import moran' finds the built extension."""
import sys
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
build_dir = os.path.join(root, 'build-python')
if os.path.isdir(build_dir):
    sys.path.insert(0, build_dir)
