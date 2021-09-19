# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import Callable, Dict, Optional, Any, Union, List, TypeVar

import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length

from fairlearn.metrics import MetricFrame

_EMPTY_BATCHES_ERR = "No data to process, please add batches with `add_data`."
_CF_BAD_STATE_ERR = ("MetricFrame expected `control_features=None`"
                     " because it was initialized as such.")

# A TypeVar to express possible batch types.
B = TypeVar('B', Dict, np.ndarray, List, pd.Series, pd.DataFrame)


class StreamingMetricFrame:
    """Streaming version of MetricFrame.

    Accumulate values to evaluate the metric(s) later on.
    See MetricFrame documentation for information on metrics.

    Parameters
    ----------
    metrics : callable or dict
        The underlying metric functions which are to be calculated. This
        can either be a single metric function or a dictionary of functions.
        These functions must be callable as
        ``fn(y_true, y_pred, **sample_params)``.
        If there are any other arguments required (such as ``beta`` for
        :func:`sklearn.metrics.fbeta_score`) then
        :func:`functools.partial` must be used.

        **Note** that the values returned by various members of the class change
        based on whether this argument is a callable or a dictionary of
        callables. This distinction remains *even if* the dictionary only
        contains a single entry.
    """

    def __init__(self,
                 *,
                 metrics: Union[Callable, Dict[str, Callable]]):
        self._metrics = metrics

        # Accumulators
        self._y_true = None
        self._y_pred = None
        self._sensitive_features = None
        self._control_features = None
        self._sample_params = None

    def reset(self):
        """Reset accumulators.

        Reset y_true, y_pred, sensitive_features, control_features and sample_params.
        """
        self._y_true = None
        self._y_pred = None
        self._sensitive_features = None
        self._control_features = None
        self._sample_params = None

    def add_data(self, y_true, y_pred, sensitive_features, control_features=None,
                 sample_params: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None):
        """Add data to the MetricFrame.

        Parameters
        ----------
        y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The ground-truth labels (for classification) or target values (for regression).

        y_pred : List, pandas.Series, numpy.ndarray, pandas.DataFrame
            The predictions.

        sensitive_features : (List, pandas.Series, dict of 1d arrays,
         numpy.ndarray, pandas.DataFrame)
            The sensitive features which should be used to create the subgroups.
            At least one sensitive feature must be provided.
            All names (whether on pandas objects or dictionary keys) must be strings.
            We also forbid DataFrames with column names of ``None``.
            For cases where no names are provided we generate names ``sensitive_feature_[n]``.

        control_features : (List, pandas.Series, dict of 1d arrays,
         numpy.ndarray, pandas.DataFrame)
            Control features are similar to sensitive features, in that they
            divide the input data into subgroups.
            Unlike the sensitive features, aggregations are not performed
            across the control features - for example, the ``overall`` property
            will have one value for each subgroup in the control feature(s),
            rather than a single value for the entire data set.
            Control features can be specified similarly to the sensitive features.
            However, their default names (if none can be identified in the
            input values) are of the format ``control_feature_[n]``.

        sample_params : dict
            Parameters for the metric function(s). If there is only one metric function,
            then this is a dictionary of strings and array-like objects, which are split
            alongside the ``y_true`` and ``y_pred`` arrays, and passed to the metric function.
            If there are multiple metric functions (passed as a dictionary), then this is
            a nested dictionary, with the first set of string keys identifying the
            metric function name, with the values being the string-to-array-like dictionaries.

            **Note** the types returned by members of the class vary based on whether
            control features are present.
        """
        check_consistent_length(y_true, y_pred)
        check_consistent_length(y_true, sensitive_features)
        self._y_true = self._concat_if_not_none(self._y_true, y_true)
        self._y_pred = self._concat_if_not_none(self._y_pred, y_pred)
        self._sensitive_features = self._concat_if_not_none(self._sensitive_features,
                                                            sensitive_features)
        if control_features is not None:
            check_consistent_length(y_true, control_features)
            self._control_features = self._concat_if_not_none(self._control_features,
                                                              control_features)
        elif self._control_features is not None:
            raise ValueError(_CF_BAD_STATE_ERR)

        if sample_params is not None:
            self._sample_params = self._concat_if_not_none(self._sample_params,
                                                           sample_params)

    def get_metric_frame(self) -> MetricFrame:
        """Get the MetricFrame computed on the accumulated values."""
        if self._y_true is None:
            raise ValueError(_EMPTY_BATCHES_ERR)
        return MetricFrame(metrics=self._metrics,
                           y_true=self._y_true,
                           y_pred=self._y_pred,
                           sensitive_features=self._sensitive_features,
                           control_features=self._control_features,
                           sample_params=self._sample_params)

    def _concat_batches(self, batches: List[B]) -> B:
        """Concatenate a list of items together.

        When merging Dicts, the values will be merged and a new dict will be created.
        DataFrames' columns need to match to be concatenated.
        Arrays, list and Series can be merged using simple functions.

        Parameters
        ----------
        batches : List[Union[Dict, nd.array, List, Serie, DataFrame]]
            A list of batch to concat together. The batches must be of the same type.

        Returns
        ---------
        Batches concatenated into a single object.
        """
        if len(batches) == 0:
            raise ValueError(_EMPTY_BATCHES_ERR)
        batch_type = type(batches[0])
        if not all([type(arr) is batch_type for arr in batches]):
            raise ValueError("Can't concatenate arrays of different types.")

        # Simple cases, array, Series and list can be concat easily.
        if batch_type is np.ndarray:
            result = np.concatenate(batches)
        elif batch_type is list:
            result = sum(batches, [])
        elif batch_type is pd.Series:
            result = pd.concat(batches)
        elif batch_type is pd.DataFrame:
            # To concat dataframe, we check if the columns match.
            col_nums = batches[0].columns
            if any([col_nums != df.columns for df in batches]):
                raise ValueError(f"Column mismatch expected {col_nums},"
                                 f" got {[df.columns for df in batches]}")
            result = pd.concat(batches)
        elif batch_type is dict:
            # Create a new dict that merges values of all keys.
            all_keys = set(sum((list(b.keys()) for b in batches), []))
            return {k: self._concat_batches([b[k] for b in batches if k in b])
                    for k in all_keys}
        else:
            raise ValueError(f"Can't concatenate {batch_type}: {batches}")
        return result

    def _concat_if_not_none(self, val1: Optional[B], val2: B) -> B:
        # Helper function that concats batches when possible.
        return val2 if val1 is None else self._concat_batches([val1, val2])
