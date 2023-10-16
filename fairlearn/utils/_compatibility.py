# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.preprocessing import OneHotEncoder
import inspect

# In sklearn 1.2, the name of the 'sparse' kwarg to OneHotEncoder
# is being changed to 'sparse_output'.

_SPARSE_OUTPUT_FALSE = {"sparse_output": False}
if "sparse_output" not in inspect.getfullargspec(OneHotEncoder).kwonlyargs:
    _SPARSE_OUTPUT_FALSE = {"sparse": False}
