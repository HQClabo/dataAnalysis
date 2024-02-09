import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dataAnalysis", 
    version="0.1",
    author="Fabian Oppliger",
    author_email="fabianoppliger@bluewin.ch",
    description="python module for data analysis with fitting and plotting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HQClabo/dataAnalysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
