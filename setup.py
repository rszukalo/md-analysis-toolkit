from setuptools import setup, find_packages

setup(
    name="mdtoolkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
    author="Ryan Szukalo",
    author_email="rszukalo@princeton.edu",
    description="A toolkit for analyzing molecular dynamics trajectories",
    keywords="molecular-dynamics, simulation, analysis",
    url="https://github.com/rszukalo/md-analysis-toolkit",
)
