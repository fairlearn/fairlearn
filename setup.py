# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import setuptools
import fairlearn

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=fairlearn.__name__,
    version=fairlearn.__version__,
    author="Miroslav Dudik, Richard Edgar, Brandon Horn, Roman Lutz, Beth Zeranski",
    author_email="fairlearn@microsoft.com",
    description="Algorithms for mitigating unfairness in supervised machine learning",
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
