# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


from sklearn.datasets import fetch_openml
import inspect

_DOWNLOAD_DIRECTORY_NAME = ".fairlearn-data"

# In sklearn 1.2, the function fetch_openml added a new keyword
# argument 'parser', and also changed the default behavior. To
# maintain compatibility, we therefore set the 'parser' argument
# to 'liac-arff', which matches the behavior before version 1.2.

if 'parser' in inspect.getfullargspec(fetch_openml).kwonlyargs:
    _PARSER_KWARG_COMPATIBILITY = {'parser': 'liac-arff'}
else:
    _PARSER_KWARG_COMPATIBILITY = {}
