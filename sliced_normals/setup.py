from setuptools import setup

setup(
    name="sliced_normals",
    version="0.2.0",
    description="Sliced Normal density estimation with cached fitting",
    author="Arindam Roy Chowdhury",
    python_requires=">=3.10,<3.13",
    packages=["sliced_normals"],
    package_dir={"sliced_normals": "."},
    package_data={"sliced_normals": ["Sample_Data.csv"]},
    install_requires=['numpy>=1.26.4,<2', 'pandas>=1.5.3,<3', 'scipy>=1.13.1,<1.14', 'autograd==1.8.0', 'pymanopt==2.2.1', 'emcee==3.1.6', 'matplotlib>=3.9,<4', 'seaborn>=0.13,<0.14', 'hyppo>=0.5,<0.6', 'joblib>=1.3,<2', 'scikit-learn>=1.3,<2', 'tqdm>=4.66,<5'],
)
