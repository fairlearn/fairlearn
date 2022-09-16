.. _intersecting_groups:

Intersecting Groups
===================

.. currentmodule:: fairlearn.metrics

The :class:`MetricFrame` class supports fairness assessment of intersecting groups in two ways:
multiple sensitive features, and control features.
Both of these can be used simultaneously.
One important point to bear in mind when performing an intersectional analysis
is that some of the intersections may have very few members (or even be empty).
This will affect the confidence interval associated with the computed metrics;
random noise has a greater effect on smaller groups.
All of these will use the following definitions:


.. doctest:: intersecting_groups_code

    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> sf_data = ['b', 'b', 'a', 'b', 'b', 'c', 'c', 'c', 'a',
    ...            'a', 'c', 'a', 'b', 'c', 'c', 'b', 'c', 'c']
    >>> from fairlearn.metrics import MetricFrame

Multiple Sensitive Features
---------------------------

Multiple sensitive features can be specified when the :class:`MetricFrame`
is constructed.
The :attr:`MetricFrame.by_group` property then holds the intersections
of these groups:

.. doctest:: intersecting_groups_code
    :options:  +NORMALIZE_WHITESPACE

    >>> import numpy as np
    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', 20)
    >>> pd.set_option('display.width', 80)
    >>> from sklearn.metrics import recall_score
    >>> g_2 = [ 8,6,8,8,8,8,6,6,6,8,6,6,6,6,8,6,6,6 ]
    >>> s_f_frame = pd.DataFrame(np.stack([sf_data, g_2], axis=1),
    ...                          columns=['SF 0', 'SF 1'])
    >>> metric_2sf = MetricFrame(metrics=recall_score,
    ...                          y_true=y_true,
    ...                          y_pred=y_pred,
    ...                          sensitive_features=s_f_frame)
    >>> metric_2sf.overall  # Same as before
    0.5
    >>> metric_2sf.by_group
    SF 0  SF 1
    a     6       0.000000
          8       1.000000
    b     6       0.666667
          8       0.500000
    c     6       0.500000
          8       0.000000
    Name: recall_score, dtype: float64

If a particular intersection of the sensitive features had no members, then
the metric would be shown as :code:`NaN` for that intersection.
Multiple metrics can also be computed at the same time:

.. doctest:: intersecting_groups_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from sklearn.metrics import precision_score
    >>> from fairlearn.metrics import count
    >>> metric_2sf_multi = MetricFrame(
    ...     metrics={'precision':precision_score,
    ...              'recall':recall_score,
    ...              'count': count},
    ...     y_true=y_true,
    ...     y_pred=y_pred,
    ...     sensitive_features=s_f_frame
    ... )
    >>> metric_2sf_multi.overall
    precision     0.6
    recall        0.5
    count        18.0
    dtype: float64
    >>> metric_2sf_multi.by_group
               precision    recall  count
    SF 0 SF 1
    a    6      0.000000  0.000000    2.0
         8      0.500000  1.000000    2.0
    b    6      1.000000  0.666667    3.0
         8      1.000000  0.500000    3.0
    c    6      0.666667  0.500000    6.0
         8      0.000000  0.000000    2.0


Control Features
----------------

Control features (sometimes called 'conditional' features) enable more detailed
fairness insights by providing a further means of splitting the data into
subgroups.
Control features are useful for cases where there is some expected variation with
a feature, so we need to compute disparities while controlling for that feature.
For example, in a loan scenario we would expect people of differing incomes to
be approved at different rates, but within each income band we would still
want to measure disparities between different sensitive features.
**However**, it should be borne in mind that due to historic discrimination, the
income band might be correlated with various sensitive features.
Because of this, control features should be used with particular caution.

When the data are split into subgroups, control features (if provided) act
similarly to sensitive features.
However, the 'overall' value for the metric is now computed for each subgroup
of the control feature(s).
Similarly, the aggregation functions (such as :func:`MetricFrame.group_max`) are
performed for each subgroup in the conditional feature(s), rather than across
them (as happens with the sensitive features).

The :class:`MetricFrame` constructor allows us to specify control features in
a manner similar to sensitive features, using a :code:`control_features=`
parameter:

.. doctest:: intersecting_groups_code
    :options:  +NORMALIZE_WHITESPACE

    >>> decision = [
    ...    0,0,0,1,1,0,1,1,0,1,
    ...    0,1,0,1,0,1,0,1,0,1,
    ...    0,1,1,0,1,1,1,1,1,0
    ... ]
    >>> prediction = [
    ...    1,1,0,1,1,0,1,0,1,0,
    ...    1,0,1,0,1,1,1,0,0,0,
    ...    1,1,1,0,0,1,1,0,0,1
    ... ]
    >>> control_feature = [
    ...    'H','L','H','L','H','L','L','H','H','L',
    ...    'L','H','H','L','L','H','L','L','H','H',
    ...    'L','H','L','L','H','H','L','L','H','L'
    ... ]
    >>> sensitive_feature = [
    ...    'A','B','B','C','C','B','A','A','B','A',
    ...    'C','B','C','A','C','C','B','B','C','A',
    ...    'B','B','C','A','B','A','B','B','A','A'
    ... ]
    >>> from sklearn.metrics import accuracy_score
    >>> metric_c_f = MetricFrame(metrics=accuracy_score,
    ...                          y_true=decision,
    ...                          y_pred=prediction,
    ...                          sensitive_features={'SF' : sensitive_feature},
    ...                          control_features={'CF' : control_feature})
    >>> # The 'overall' property is now split based on the control feature
    >>> metric_c_f.overall
    CF
    H    0.4285...
    L    0.375...
    Name: accuracy_score, dtype: float64
    >>> # The 'by_group' property looks similar to how it would if we had two sensitive features
    >>> metric_c_f.by_group
    CF  SF
    H   A     0.2...
        B     0.4...
        C     0.75...
    L   A     0.4...
        B     0.2857...
        C     0.5...
    Name: accuracy_score, dtype: float64

Note how the :attr:`MetricFrame.overall` property is stratified based on the
supplied control feature. The :attr:`MetricFrame.by_group` property allows
us to see disparities between the groups in the sensitive feature for each
group in the control feature.
When displayed like this, :attr:`MetricFrame.by_group` looks similar to
how it would if we had specified two sensitive features (although the
control features will always be at the top level of the hierarchy).

With the :class:`MetricFrame` computed, we can perform aggregations:

.. doctest:: intersecting_groups_code
    :options:  +NORMALIZE_WHITESPACE

    >>> # See the maximum accuracy for each value of the control feature
    >>> metric_c_f.group_max()
    CF
    H    0.75
    L    0.50
    Name: accuracy_score, dtype: float64
    >>> # See the maximum difference in accuracy for each value of the control feature
    >>> metric_c_f.difference(method='between_groups')
    CF
    H    0.55...
    L    0.2142...
    Name: accuracy_score, dtype: float64

In each case, rather than a single scalar, we receive one result for each
subgroup identified by the conditional feature. The call
:code:`metric_c_f.group_max()` call shows the maximum value of the metric across
the subgroups of the sensitive feature within each value of the control feature.
Similarly, :code:`metric_c_f.difference(method='between_groups')` call shows the
maximum difference between the subgroups of the sensitive feature within
each value of the control feature.
For more examples, please
see the :ref:`sphx_glr_auto_examples_plot_new_metrics.py` notebook in the
:ref:`examples`.

Finally, a :class:`MetricFrame` can use multiple control features, multiple
sensitive features and multiple metric functions simultaneously.
