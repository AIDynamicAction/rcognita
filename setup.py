import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LearnRLSK", # Replace with your own username
    version="1.0.2",
    author="Pavel Osinenko",
    author_email="p.osinenko@skoltech.ru",
    description="Learning Reinforcement Learning - Skoltech",
    long_description="See homepage",
    long_description_content_type="text/markdown",
    url="https://github.com/OsinenkoP/learnRL-py",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "control",
        "slycot",
        "future",
        "tabulate",
        "mpldatacursor",
        "svgpath2mpl",
        "nbconvert",
        "cmake",
        "scikit-build"],
    dependency_links=['https://github.com/CPCLAB-UNIPI/SIPPY/tarball/master#egg=SIPPY']
)