from setuptools import setup, find_packages
from os.path import join, dirname
import sys, os

sys.path.insert(0, os.path.abspath(__file__ + '/..'))
from rcognita import __version__
# import rcognita; print(rcognita.__file__)

setup(
    name='rcognita',
    version=__version__,
    author="AIDynamicAction",
    description="rcognita is a framework for hybrid agent-environment loop simulation, with a library of predictive and stabilizing reinforcement learning setups",
    url="https://github.com/AIDynamicAction/rcognita",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "matplotlib >= 3.2.2",
        "mpldatacursor == 0.7.1",
        "numpy >= 1.20.1",
        "scipy >= 1.5.0",
        "svgpath2mpl == 0.2.1",
        "tabulate == 0.8.7",
        "torch >= 1.6.0",
        "systems == 0.1.0",
        "shapely == 1.7.1"],
    python_requires=">=3.6",
)