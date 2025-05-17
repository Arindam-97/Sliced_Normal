# Sliced_Normal/setup.py

from setuptools import setup, find_packages

setup(
    name="sliced_normals",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pymanopt",
        "autograd",
        "emcee",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ],
    author="Arindam Roy Chowdhury",
    description="Package for sliced normal density estimation and sampling",
)
