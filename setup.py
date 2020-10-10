import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rcognita",
    version="1.7.6",
    author="Pavel Osinenko, Eli Bolotin",
    author_email="p.osinenko@skoltech.ru, ebolotin6.git@gmail.com",
    description="Python framework for Dynamic RL",
    long_description="See url",
    long_description_content_type="text/markdown",
    url="https://github.com/AIDA-Skoltech/rcognita",
    packages=['rcognita'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'rcogn = rcognita.__main__:main',
            'rcognt = rcognita.scripts.ctest:main'
        ],
    },
    python_requires="~=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "control",
        "slycot",
        "bleach",
        "docutils",
        "Pygments",
        "future",
        "tabulate",
        "mpldatacursor",
        "svgpath2mpl",
        "nbconvert",
        "cmake",
        "scikit-build",
        "colorama",
        "pkginfo",
        "requests",
        "tqdm"],
)