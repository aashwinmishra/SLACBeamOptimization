from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.17", "scipy>=1.4.1", "matplotlib>=3.2.1", "jupyter>=1.0.0", "scikit-image>=0.16.2",
                "xrt"]

setup(
    name="lcls_beamline_optimization",
    version="0.0.2",
    author="AAM",
    author_email="--",
    description="Tools for LCLS beamline Optimization",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="-",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
)
