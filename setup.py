import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="learnRL-py", # Replace with your own username
    version="1.6.0",
    author="Pavel Osinenko",
    author_email="p.osinenko@skoltech.ru",
    description="learnRL-py",
    long_description="See homepage",
    long_description_content_type="text/markdown",
    url="https://github.com/AIDA-Skoltech/learnRL-py",
    packages=['learnRL-py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'lrl = learnRL-py.__main__:main',
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