
from setuptools import setup, find_packages

setup(
    name="PyPhysTree",
    version="1.0.0",
    description="Physics-Informed Continuous MCTS for High-Dimensional Design",
    author="Suvo Banik",
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn', 'pyDOE', 'scipy', 'setuptools'],
    python_requires='>=3.8',
)
