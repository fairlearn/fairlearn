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

# Fetch separate requirements file for each extension
extras = ["customplots"]
extras_require = dict()
for e in extras:
    req_file = "requirements-{0}.txt".format(e)
    with open(req_file) as f:
        extras_require[e] = [line.strip() for line in f]

setuptools.setup(
    name=fairlearn.__name__,
    version=fairlearn.__version__,
    author="Miroslav Dudik, Richard Edgar, Brandon Horn, Roman Lutz",
    author_email="fairlearn@microsoft.com",
    description="Algorithms for mitigating unfairness in supervised machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairlearn/fairlearn",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
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
