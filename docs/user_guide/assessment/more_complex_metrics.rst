.. _more_complex_metrics:

More Complex Metrics
====================

.. currentmodule:: fairlearn.metrics

So far, we have stuck to relatively simple cases, where the inputs are 1-D vectors of scalars,
and the metric functions return scalar values.
However, this need not be the case - we noted above that we were going to be vague as to the
contents of the input vectors and the return value of the metric function.

Non-Scalar Results from Metric Functions
----------------------------------------

Metric functions need not return a scalar value.
A straightforward example of this is the confusion matrix.
Such return values are fully supported by :class:`MetricFrame`:


.. doctest:: assessment_metrics
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

.. doctest:: assessment_metrics
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
----------------------------

:class:`MetricFrame` can also handle cases when the :math:`Y_{true}` and/or :math:`Y_{pred}` vectors
are not vectors of scalars.
It is the metric function(s) which gives meaning to these values - :class:`MetricFrame` itself
just slices the vectors up according to the sensitive feature(s) and the control feature(s).

As a toy example, suppose that our ``y`` values (both true and predicted) are tuples representing
the dimensions of a rectangle.
For some reason known only to our fevered imagination (although it might possibly be due to a
desire for a *really* simple example), we are interested in the areas of these rectangles.
In particular, we want to calculate the mean of the area ratios. That is:


.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

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

.. doctest:: assessment_metrics
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

