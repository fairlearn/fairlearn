.. _advanced_metricframe:

Advanced Usage of MetricFrame
=============================

.. currentmodule:: fairlearn.metrics

In this section, we will discuss how :class:`MetricFrame` can
be used in more sophisticated scenarios.
All code examples will use the following definitions:


.. doctest:: advanced_metricframe_code

    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> sf_data = ['b', 'b', 'a', 'b', 'b', 'c', 'c', 'c', 'a',
    ...            'a', 'c', 'a', 'b', 'c', 'c', 'b', 'c', 'c']
    >>> from fairlearn.metrics import MetricFrame



.. _extra_arguments_metric_functions:

Extra Arguments to Metric functions
-----------------------------------

The metric functions supplied to :class:`MetricFrame` might require additional
arguments.
These fall into two categories: 'scalar' arguments (which affect the operation
of the metric function), and 'per-sample' arguments (such as sample weights).
Different approaches are required to use each of these.

Scalar Arguments
^^^^^^^^^^^^^^^^

We do not directly support scalar arguments for the metric functions.
If these are required, then use :func:`functools.partial` to prebind the
required arguments to the metric function:

.. doctest:: advanced_metricframe_code
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
^^^^^^^^^^^^^^^^^^^^

If there are per-sample arguments (such as sample weights), these can also be 
provided in a dictionary via the ``sample_params`` argument.
The keys of this dictionary are the argument names, and the values are 1-D
arrays equal in length to ``y_true`` etc.:

.. doctest:: advanced_metricframe_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from sklearn.metrics import recall_score
    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', 20)
    >>> pd.set_option('display.width', 80)
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

.. doctest:: advanced_metricframe_code
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


No `y_true` or `y_pred`
^^^^^^^^^^^^^^^^^^^^^^^

In some cases, a metric may not have `y_true` or `y_pred` arguments, or even
either of them.
One example of this is the selection rate metric, which only considers
the `y_pred` values (selection rate is used when computing
:ref:`demographic parity <assessment_demographic_parity>`).
However, :class:`MetricFrame` requires all supplied metric functions to
conform to the scikit-learn metric paradigm, where the first two arguments
to the metric function are the `y_true` and `y_pred` arrays.
The workaround in this case is to supply a dummy argument.
This is the approach we use in :meth:`selection_rate`, which simply ignores
the supplied `y_true` argument.
When invoking `MetricFrame`, a `y_true` array of the appropriate length
must still be supplied.
For example:

.. doctest:: advanced_metricframe_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import selection_rate
    >>> dummy_y_true = [x for x in range(len(y_pred))]
    >>> sel_rate_frame = MetricFrame(metrics=selection_rate,
    ...                              y_true=dummy_y_true,
    ...                              y_pred=y_pred,
    ...                              sensitive_features=pd.Series(sf_data, name='SF 0'))
    >>> sel_rate_frame.overall
    0.55555...
    >>> sel_rate_frame.by_group
    SF 0
    a    0.75
    b    0.50
    c    0.50
    Name: selection_rate, dtype: float64


.. _more_complex_metrics:

More Complex Metrics
--------------------


Metric functions often return a single scalar value based on arguments which are vectors of
scalars.
This is how :class:`MetricFrame` was introduced in the :ref:`perform_fairness_assessment`
section above.
However, this need not be the case - indeed, we were rather vague about the
contents of the input vectors and the return value of the metric function.
We will now show how to use :class:`MetricFrame` in cases where the result is not
a scalar, and when the inputs are not vectors of scalars.

Non-Scalar Results from Metric Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metric functions need not return a scalar value.
A straightforward example of this is the confusion matrix.
Such return values are fully supported by :class:`MetricFrame`:


.. doctest:: advanced_metricframe_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from sklearn.metrics import confusion_matrix
    >>> mf_conf = MetricFrame(
    ...    metrics=confusion_matrix,
    ...    y_true=y_true,
    ...    y_pred=y_pred,
    ...    sensitive_features=sf_data
    ... )
    >>> mf_conf.overall
    array([[2, 4],
           [6, 6]]...)
    >>> mf_conf.by_group
    sensitive_feature_0
    a    [[0, 2], [1, 1]]
    b    [[1, 0], [2, 3]]
    c    [[1, 2], [3, 2]]
    Name: confusion_matrix, dtype: object

Obviously for such cases, operations such as :meth:`MetricFrame.difference` have no meaning.
However, if scalar-returning metrics are also present, they will still be calculated:

.. doctest:: advanced_metricframe_code
    :options:  +NORMALIZE_WHITESPACE

    >>> mf_conf_recall = MetricFrame(
    ...    metrics={ 'conf_mat':confusion_matrix, 'recall':recall_score },
    ...    y_true=y_true,
    ...    y_pred=y_pred,
    ...    sensitive_features=sf_data
    ... )
    >>> mf_conf_recall.overall
    conf_mat    [[2, 4], [6, 6]]
    recall                   0.5
    dtype: object
    >>> mf_conf_recall.by_group
                                 conf_mat  recall
    sensitive_feature_0
    a                    [[0, 2], [1, 1]]     0.5
    b                    [[1, 0], [2, 3]]     0.6
    c                    [[1, 2], [3, 2]]     0.4
    >>> mf_conf_recall.difference()
    conf_mat    NaN
    recall      0.2
    dtype: float64

We see that the difference between group recall scores has been calculated, while a value of
:code:`None` has been returned for the meaningless 'maximum difference between two confusion matrices'
entry.

Inputs are Arrays of Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`MetricFrame` can also handle cases when the :math:`Y_{true}` and/or :math:`Y_{pred}` vectors
are not vectors of scalars.
It is the metric function(s) which gives meaning to these values - :class:`MetricFrame` itself
just slices the vectors up according to the sensitive feature(s) and the control feature(s).

As a toy example, suppose that our ``y`` values (both true and predicted) are tuples representing
the dimensions of a rectangle.
For some reason known only to our fevered imagination (although it might possibly be due to a
desire for a *really* simple example), we are interested in the areas of these rectangles.
In particular, we want to calculate the mean of the area ratios. That is:


.. doctest:: advanced_metricframe_code
    :options:  +NORMALIZE_WHITESPACE

    >>> import numpy as np
    >>> def area_metric(y_true, y_pred):
    ...     def calc_area(a):
    ...         return a[0] * a[1]
    ...
    ...     y_ts = np.asarray([calc_area(x) for x in y_true])
    ...     y_ps = np.asarray([calc_area(x) for x in y_pred])
    ...
    ...     return np.mean(y_ts / y_ps)


This is a perfectly good metric for :class:`MetricFrame`, provided we supply appropriate
inputs.

.. doctest:: advanced_metricframe_code
    :options:  +NORMALIZE_WHITESPACE

    >>> y_rect_true = [(4,9), (3,8), (2,10)]
    >>> y_rect_pred = [(1,12), (2,1), (5, 2)]
    >>> rect_groups = { 'sf_0':['a', 'a', 'b'] }
    >>>
    >>> mf_non_scalar = MetricFrame(
    ...      metrics=area_metric,
    ...      y_true=y_rect_true,
    ...      y_pred=y_rect_pred,
    ...      sensitive_features=rect_groups  
    ... )
    >>> print(mf_non_scalar.overall)
    5.6666...
    >>> print(mf_non_scalar.by_group)
    sf_0
    a    7.5
    b    2.0
    Name: area_metric, dtype: float64

For a more concrete example, consider an image recognition algorithm which draws a bounding box
around some region of interest.
We will want to compare the 'true' bounding boxes (perhaps from human annotators) with the
ones predicted by our model.
A straightforward metric for this purpose is the IoU or 'intersection over union.'
As the name implies, this metric takes two rectangles, and computes the area of their intersection
and divides it by the area of their union.
If the two rectangles are disjoint, then the IoU will be zero.
If the two rectangles are identical, then the IoU will be one.
This is presented in full in our 
`example notebook <../auto_examples/plot_metricframe_beyond_binary_classification.html#non-scalar-inputs>`_.

