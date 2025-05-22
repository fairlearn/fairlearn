# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import narwhals.stable.v1 as nw

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
    or generated from a base and index.

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

    name : str | None
        Optional name for the feature
    """

    def __init__(self, base_name: str, feature_vector, index: int, name: str | None = None):
        """Help with the metrics."""
        nw_feature_vector = nw.from_native(feature_vector, pass_through=True, allow_series=True)
        is_nw_series = isinstance(nw_feature_vector, nw.Series)

        self.raw_feature_ = (
            list(nw_feature_vector) if not is_nw_series else nw_feature_vector.to_list()
        )

        self.name_ = "{0}{1}".format(base_name, index)
        if name is not None:
            self.name_ = name
        elif is_nw_series:
            if (name_ := nw_feature_vector.name) is not None:
                if isinstance(name_, str):
                    self.name_ = name_
                else:
                    msg = _SERIES_NAME_NOT_STRING.format(name_, type(name_))
                    raise ValueError(msg)
