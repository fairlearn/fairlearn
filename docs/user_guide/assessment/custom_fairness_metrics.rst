
.. _custom_fairness_metrics:

Defining custom fairness metrics
================================

.. currentmodule:: fairlearn.metrics

Higher level machine learning algorithms (such as hyperparameter tuners) often
make use of metric functions to guide their optimisations.
Such algorithms generally work with scalar results, so if we want the tuning
to be done on the basis of our fairness metrics, we need to perform aggregations
over the :class:`MetricFrame`.

We provide a convenience function, :func:`fairlearn.metrics.make_derived_metric`,
to generate scalar-producing metric functions based on the aggregation methods
mentioned above (:meth:`MetricFrame.group_min`, :meth:`MetricFrame.group_max`,
:meth:`MetricFrame.difference`, and :meth:`MetricFrame.ratio`).
This takes an underlying metric function, the name of the desired transformation, and
optionally a list of parameter names which should be treated as sample aligned parameters
(such as `sample_weight`).
The result is a function which builds the :class:`MetricFrame` internally and performs
the requested aggregation. For example:

.. doctest:: custom_fairness_metrics_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import make_derived_metric, MetricFrame
    >>> from sklearn.metrics import recall_score
    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> sf_data = ['b', 'b', 'a', 'b', 'b', 'c', 'c', 'c', 'a',
    ...            'a', 'c', 'a', 'b', 'c', 'c', 'b', 'c', 'c']
    >>> recall_difference = make_derived_metric(metric=recall_score,
    ...                                        transform='difference')
    >>> recall_difference(y_true, y_pred,
    ...                   sensitive_features=sf_data)
    0.19999...
    >>> MetricFrame(metrics=recall_score,
    ...             y_true=y_true,
    ...             y_pred=y_pred,
    ...             sensitive_features=sf_data).difference()
    0.19999...

We use :func:`fairlearn.metrics.make_derived_metric` to manufacture a number
of such functions which are commonly used.
The table below displays the aggregations that we have created for each
base metric:

=============================================== ================= ================= ================== =============
Base metric                                     :code:`group_min` :code:`group_max` :code:`difference` :code:`ratio`
=============================================== ================= ================= ================== =============
:func:`.false_negative_rate`                    .                 .                 Y                  Y
:func:`.false_positive_rate`                    .                 .                 Y                  Y
:func:`.selection_rate`                         .                 .                 Y                  Y
:func:`.true_negative_rate`                     .                 .                 Y                  Y
:func:`.true_positive_rate`                     .                 .                 Y                  Y
:func:`sklearn.metrics.accuracy_score`          Y                 .                 Y                  Y
:func:`sklearn.metrics.balanced_accuracy_score` Y                 .                 .                  .
:func:`sklearn.metrics.f1_score`                Y                 .                 .                  .
:func:`sklearn.metrics.log_loss`                .                 Y                 .                  .
:func:`sklearn.metrics.mean_absolute_error`     .                 Y                 .                  .
:func:`sklearn.metrics.mean_squared_error`      .                 Y                 .                  .
:func:`sklearn.metrics.precision_score`         Y                 .                 .                  .
:func:`sklearn.metrics.r2_score`                Y                 .                 .                  .
:func:`sklearn.metrics.recall_score`            Y                 .                 .                  .
:func:`sklearn.metrics.roc_auc_score`           Y                 .                 .                  .
:func:`sklearn.metrics.zero_one_loss`           .                 Y                 Y                  Y
=============================================== ================= ================= ================== =============

The names of the generated functions are of the form
:code:`fairlearn.metrics.<base_metric>_<transformation>`.
For example :code:`fairlearn.metrics.accuracy_score_difference` and
:code:`fairlearn.metrics.precision_score_group_min`.
