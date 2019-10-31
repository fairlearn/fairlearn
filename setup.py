# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import setuptools
import fairlearn

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=fairlearn.__name__,
    version=fairlearn.__version__,
    author="Miroslav Dudik",
    author_email="mdudik@gmail.com",
    description="Algorithms for fair classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairlearn/fairlearn",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=[
        "matplotlib>=3.0.3",
        "numpy>=1.17.2",
        "pandas>=0.25.1",
        "scikit-learn>=0.21.3",
        "scipy>=1.3.1"
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    zip_safe=False,
)
