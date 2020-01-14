# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

_VERSION = "0.4.1"

setuptools.setup(
    name='fairlearn',
    version=_VERSION,
    author="Miroslav Dudik, Richard Edgar, Brandon Horn, Roman Lutz",
    author_email="fairlearn@microsoft.com",
    description="Algorithms for mitigating unfairness in supervised machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairlearn/fairlearn-full",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=[
        "fairlearn-core[widget,customplots]>={}".format(_VERSION)
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
)
