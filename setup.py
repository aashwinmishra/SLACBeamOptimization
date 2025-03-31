from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.17", "scipy>=1.4.1", "matplotlib>=3.2.1", "jupyter>=1.0.0", "scikit-image>=0.16.2",
                "xrt", "xopt]

setup(
    name="lcls_beamline_toolbox",
    version="0.0.1",
    author="",
    author_email="",
    description="tools for LCLS beamline calculations",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/aashwinmishra/SLACBeamOptimization",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
)
