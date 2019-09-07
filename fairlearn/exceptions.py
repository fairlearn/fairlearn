# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class NotFittedException(ValueError):
    """ Exception to use if predict is called before fit.
    """
