"""Legacy installer entry point. Prefer `python -m pip install -e .` at the root."""
from pathlib import Path
import os
from setuptools import setup
os.chdir(Path(__file__).resolve().parents[1])
setup()
