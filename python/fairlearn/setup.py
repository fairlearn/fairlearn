# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from setuptools import setup, find_packages
import fairlearn

with open(os.path.join('..', '..', "README.md"), "r") as fh:
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
        "ipywidgets>=7.5.0",
        "fairlearn-core>=0.4.0"
    ],
    classifiers=(
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ),
    include_package_data=True,
    data_files=[
        ('share/jupyter/nbextensions/fairlearn-widget', [
            'fairlearn/widget/static/extension.js',
            'fairlearn/widget/static/extension.js.map',
            'fairlearn/widget/static/index.js',
            'fairlearn/widget/static/index.js.map'
        ]),
        ("etc/jupyter/nbconfig/notebook.d", [
            "jupyter-config/nbconfig/notebook.d/fairlearn-widget.json"
        ]),
        ('share/jupyter/lab/extensions', [
            'fairlearn/widget/js/'
            'fairlearn_widget/labextension/fairlearn-widget-0.1.0.tgz'
        ])],
    zip_safe=False,
)
