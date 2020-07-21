import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LearnRLSK", # Replace with your own username
    version="1.1.0",
    author="Pavel Osinenko",
    author_email="p.osinenko@skoltech.ru",
    description="Learning Reinforcement Learning - Skoltech",
    long_description="See homepage",
    long_description_content_type="text/markdown",
    url="https://github.com/OsinenkoP/LearnRLSK",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'lrl = LearnRLSK.__main__:main',
        ],
    },
    python_requires="~=3.7",
    install_requires=[
        "sippy @ https://github.com/ebolotin6/SIPPY/archive/master.zip"
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