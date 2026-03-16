"""Add python/ to sys.path so 'import moran' finds the package + built extension."""
import sys
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'python'))
