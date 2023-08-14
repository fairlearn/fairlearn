# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import setuptools

import fairlearn

# Fetch ReadMe
with open("README.rst", "r") as fh:
    long_description = fh.read()

# Use requirements.txt to set the install_requires
with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f]

setuptools.setup(
    name=fairlearn.__name__,
    version=fairlearn.__version__,
    author=(
        "Miroslav Dudik, Richard Edgar, Adrin Jalali, Roman Lutz, Michael Madaio, Hilde"
        " Weerts"
    ),
    author_email="fairlearn-internal@python.org",
    description=(
        "A Python package to assess and improve fairness of machine learning models."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairlearn/fairlearn",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    include_package_data=True,
    zip_safe=False,
)
