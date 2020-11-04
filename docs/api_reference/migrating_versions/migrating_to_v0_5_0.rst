.. _migrating_to_v0_5_0:

Migrating to v0.5.0 from v0.4.6
===============================

The update from v0.4.6 of Fairlearn has brought some major changes. This
document goes through the adjustments required.

Metrics
-------

We have substantially altered the :mod:`fairlearn.metrics` module.
In place of calling ``group_summary()`` to produce a :py:class:`sklearn.utils.Bunch`
containing the disaggregated metrics, we have a new class, :class:`.MetricFrame`.
The key advantages of the new API are:

    - Support for evalulating multiple metric functions at once
    - Support for multiple sensitive features
    - Support for control features

The :class:`MetricFrame` class has a constructor similar to ``group_summary()``.
In v0.4.6, one would write

.. code-block::

    gs = group_summary(metric_func, y_true, y_pred, sensitive_features=A_col)

With the new API, this becomes

.. code-block::

    mf = MetricFrame(metric_func, y_true, y_pred, sensitive_features=A_col)

The new object has :attr:`.MetricFrame.overall` and :attr:`.MetricFrame.by_group`
properties, to access the metric evaluated on the entire dataset, and the metric
evaluated on the subgroups of ``A_col``.

In v0.4.6, we provided the following aggregator functions to compute a single scalar
from the result of ``group_summary()``.

    - ``group_min_from_summary()``
    - ``group_max_from_summary()``
    - ``difference_from_summary()``
    - ``ratio_from_summary()``

With :class:`.MetricFrame` these become methods:

    - :meth:`.MetricFrame.group_min`
    - :meth:`.MetricFrame.group_max`
    - :meth:`.MetricFrame.difference`
    - :meth:`.MetricFrame.ratio`

Before, one might write:

.. code-block::

    min_by_group = group_min_from_summary(gs)

Now, one can write:

.. code-block::

    min_by_group = mf.group_min()

There is a ``method=`` argument to :meth:`.MetricFrame.difference`
and :meth:`.MetricFrame.ratio` which can be set to ``between_groups``
and ``to_overall``. To obtain behaviour similar to:

.. code-block::

    diff = difference_from_summary(gs)

use

.. code-block::

    diff = mf.difference(method='between_groups')

The ``to_overall`` alternative will evaluate the differences (or ratios)
relative to the overall value, rather than just between the groups identified
by the sensitive feature.

The ``make_derived_metric()`` function has been removed, but will be reintroduced
in a future release. The pregenerated functions such as ``accuracy_score_group_min()``
and ``precision_score_difference()`` remain.

For an introduction to all the new features, see the 
:ref:`sphx_glr_auto_examples_plot_new_metrics.py` example in
:ref:`sphx_glr_auto_examples`.


Renaming of members
-------------------

We have renamed a number of class members from ``_<name>`` to ``<name>_``.
For example in both :class:`.ExponentiatedGradient` and :class:`.GridSearch`,
the ``_predictors`` member is now called ``predictors_``.


Exponentiated Gradient and Moments
----------------------------------

In addition to the trailing underscore change mentioned above, several
adjustments have been made to :class:`.ExponentiatedGradient`.
The ``T`` argument has been renamed to ``max_iter``, and the ``eta_mul``
argument to ``eta0``.

Furthermore, the ``eps`` argument was previously being used for two
different purposes, and this has now been refined.
The ``eps`` argument itself is now solely used to set the L1 norm
bound used to control the excess constraint violation (beyond that
allowed by the constraint object itself).
The usage of ``eps`` as the righthand side of the constraints
has now been moved to the :class:`.Moment` classes.

For classification moments, ``ConditionalSelectionRate`` has been
renamed to :class:`.UtilityParity`, and there are three new
constructor arguments: ``difference_bound``, ``ratio_bound`` (which
replaces ``ratio``) and ``ratio_bound_slack``.

For regression moments, :class:`.ConditionalLossMoment` and its
subclasses have gained a new argument ``upper_bound`` to serve as
the righthand side of the constraints.

Several :class:`.Moment` objects have also been renamed in an effort
to improve consistency:

    - ``ErrorRateRatio`` has become :class:`.ErrorRateParity` (when used
      with the ``ratio_bound`` and ``ratio_bound_slack`` arguments)
    - ``TruePositiveRateDifference`` has become :class:`.TruePositiveRateParity`
      (when used with the ``difference_bound`` argument)
    - ``ConditionalSelectionRate`` has become :class:`.UtilityParity`
    - ``GroupLossMoment`` has become :class:`.BoundedGroupLoss`
    - ``AverageLossMoment`` has become :class:`.MeanLoss`