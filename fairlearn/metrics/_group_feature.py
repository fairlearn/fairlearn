# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import Optional

import numpy as np
import pandas as pd

_SERIES_NAME_NOT_STRING = "Series name must be a string. Value '{0}' was of type {1}"


class GroupFeature:
    """A helper class for Fairlearn metrics.

    Each instance of this class holds a single 'grouping' feature
    on behalf of a metric object. A grouping feature is either
    a sensitive or conditional feature which is used to divide
    up the sample population.

    Note that if a Pandas Series is provided then its name must either
    be ``None`` or a string. This is to prevent integer 'names' leaking out
    which then make calls which accept names or indices ambiguous (think
    `unstack()` and `groupby()`).

    This class takes care of encoding the grouping feature, and producing
    masks (used to select samples from the population) corresponding
    to the subgroups it identifies.

    It also holds the feature name, which can be supplied by the caller,
    or generated from a base and and index.

    Parameters
    ----------
    base_name : str
        The base string to use as a feature name if `name` is not specified.
        The value of `index` is appended

    feature_vector : array_like
        Some sort of array encoding the feature. It is fed into
        :class:`sklrearn.preprocessing.LabelEncoder` for easy masking

    index : int
        Used together with `base_name` when automatically generating a name

    name : str
        Optional name for the feature
    """

    def __init__(self, base_name: str, feature_vector, index: int, name: Optional[str]):
        """Help with the metrics."""
        self.classes_ = np.unique(feature_vector)
        self.raw_feature_ = feature_vector

        self.name_ = "{0}{1}".format(base_name, index)
        if name is not None:
            self.name_ = name
        elif isinstance(feature_vector, pd.Series):
            if feature_vector.name is not None:
                if isinstance(feature_vector.name, str):
                    self.name_ = feature_vector.name
                else:
                    msg = _SERIES_NAME_NOT_STRING.format(
                        feature_vector.name, type(feature_vector.name)
                    )
                    raise ValueError(msg)
