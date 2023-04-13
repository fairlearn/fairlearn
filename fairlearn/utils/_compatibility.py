# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import inspect

# In sklearn 1.2, the function fetch_openml added a new keyword
# argument 'parser', and also changed the default behavior. To
# maintain compatibility, we therefore set the 'parser' argument
# to 'liac-arff', which matches the behavior before version 1.2.

_PARSER_KWARG = {"parser": "liac-arff"}
if "parser" not in inspect.getfullargspec(fetch_openml).kwonlyargs:
    _PARSER_KWARG = {}

# In sklearn 1.2, the name of the 'sparse' kwarg to OneHotEncoder
# is being changed to 'sparse_output'.

_SPARSE_OUTPUT_FALSE = {"sparse_output": False}
if "sparse_output" not in inspect.getfullargspec(OneHotEncoder).kwonlyargs:
    _SPARSE_OUTPUT_FALSE = {"sparse": False}
