import os
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

with open(req_path, encoding="utf-8") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="sfcv",
    version="0.1.1",
    description="Step Forward Cross Validation for Bioactivity Prediction",
    author="Manas Mahale",
    author_email="manas.m.mahale@gmail.com",
    license="Apache 2.0",
    packages=find_packages(),
    package_data={"sfcv": ["logdmodel.txt"]},
    install_requires=install_requires,
    python_requires=">=3.11",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
