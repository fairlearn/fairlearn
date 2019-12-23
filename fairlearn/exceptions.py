# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module holding specialised exceptions for fairlearn."""


class NotFittedException(ValueError):
    """Exception to use if :code:`predict()` is called before :code:`fit()`."""
