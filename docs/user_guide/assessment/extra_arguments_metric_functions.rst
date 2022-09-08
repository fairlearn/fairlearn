.. _extra_arguments_metric_functions:

Extra Arguments to Metric functions
===================================

.. currentmodule:: fairlearn.metrics

The metric functions supplied to :class:`MetricFrame` might require additional
arguments.
These fall into two categories: 'scalar' arguments (which affect the operation
of the metric function), and 'per-sample' arguments (such as sample weights).
Different approaches are required to use each of these.

Scalar Arguments
----------------

We do not directly support scalar arguments for the metric functions.
If these are required, then use :func:`functools.partial` to prebind the
required arguments to the metric function:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> import functools
    >>> from sklearn.metrics import fbeta_score
    >>> fbeta_06 = functools.partial(fbeta_score, beta=0.6)
    >>> metric_beta = MetricFrame(metrics=fbeta_06,
    ...                           y_true=y_true,
    ...                           y_pred=y_pred,
    ...                           sensitive_features=sf_data)
    >>> metric_beta.overall
    0.56983...
    >>> metric_beta.by_group
    sensitive_feature_0
    a    0.365591
    b    0.850000
    c    0.468966
    Name: metric, dtype: float64


Per-Sample Arguments
--------------------

If there are per-sample arguments (such as sample weights), these can also be 
provided in a dictionary via the ``sample_params`` argument.
The keys of this dictionary are the argument names, and the values are 1-D
arrays equal in length to ``y_true`` etc.:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> s_w = [1, 2, 1, 3, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 1, 1]
    >>> s_p = { 'sample_weight':s_w }
    >>> weighted = MetricFrame(metrics=recall_score,
    ...                        y_true=y_true,
    ...                        y_pred=y_pred,
    ...                        sensitive_features=pd.Series(sf_data, name='SF 0'),
    ...                        sample_params=s_p)
    >>> weighted.overall
    0.45...
    >>> weighted.by_group
    SF 0
    a    0.500000
    b    0.583333
    c    0.250000
    Name: recall_score, dtype: float64

If multiple metrics are being evaluated, then ``sample_params`` becomes a 
dictionary of dictionaries.
The first key to this dictionary is the name of the metric as specified
in the ``metrics`` argument.
The keys of the inner dictionary are the argument names, and the values
are the 1-D arrays of sample parameters for that metric.
For example:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> s_w_2 = [3, 1, 2, 3, 2, 3, 1, 4, 1, 2, 3, 1, 2, 1, 4, 2, 2, 3]
    >>> metrics = {
    ...    'recall' : recall_score,
    ...    'recall_weighted' : recall_score,
    ...    'recall_weight_2' : recall_score
    ... }
    >>> s_p = {
    ...     'recall_weighted' : { 'sample_weight':s_w },
    ...     'recall_weight_2' : { 'sample_weight':s_w_2 }
    ... }
    >>> weighted = MetricFrame(metrics=metrics,
    ...                        y_true=y_true,
    ...                        y_pred=y_pred,
    ...                        sensitive_features=pd.Series(sf_data, name='SF 0'),
    ...                        sample_params=s_p)
    >>> weighted.overall
    recall             0.500000
    recall_weighted    0.454545
    recall_weight_2    0.458333
    dtype: float64
    >>> weighted.by_group
          recall  recall_weighted  recall_weight_2
    SF 0
    a        0.5         0.500000         0.666667
    b        0.6         0.583333         0.600000
    c        0.4         0.250000         0.272727

Note that there is no concept of a 'global' sample parameter (e.g. a set
of sample weights to be applied for all metric functions).
In such a case, the sample parameter in question must be repeated in
the nested dictionary for each metric function.