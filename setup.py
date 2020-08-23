import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rcognita", # Replace with your own username
    version="1.5.1",
    author="Pavel Osinenko, Eli Bolotin",
    author_email="p.osinenko@skoltech.ru, ebolotin6.git@gmail.com",
    description="Python framework for Dynamic RL",
    long_description="See homepage",
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
            'rcognita = rcognita.__main__:main',
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