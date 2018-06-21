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
    url="https://github.com/Microsoft/fairlearn",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "pandas"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
