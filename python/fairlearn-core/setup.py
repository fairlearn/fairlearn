# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from setuptools import setup, find_packages
import fairlearn

# README lives in repository root
if os.getcwd() == os.path.dirname(os.path.realpath(__file__)):
    # look two levels up for README
    readme_directory = os.path.join("..", "..")
elif os.path.commonprefix([os.getcwd(), os.path.dirname(os.path.realpath(__file__))]) == os.getcwd():
    readme_directory = os.getcwd()
else:
    raise Exception("Cannot locate README file. Please execute setup.py from repository root.")

with open(os.path.join(readme_directory, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name=fairlearn.__name__,
    version=fairlearn.__version__,
    author="Miroslav Dudik, Richard Edgar, Brandon Horn, Roman Lutz",
    author_email="fairlearn@microsoft.com",
    description="Algorithms for mitigating unfairness in supervised machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairlearn/fairlearn",
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        "numpy>=1.17.2",
        "pandas>=0.25.1",
        "scikit-learn>=0.21.3",
        "scipy>=1.3.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    include_package_data=True,
    zip_safe=False,
    extras_require={
        "postprocessingplot": [
            "matplotlib>=3.0.3"
        ]
    }
)
