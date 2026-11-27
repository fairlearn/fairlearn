.. _prejudice_remover:

Prejudice Remover
=================

.. currentmodule:: fairlearn.linear_model

Fairlearn provides the prejudice remover of :footcite:t:`kamishima2012prejudice`
as :class:`PrejudiceRemover`.
It is an in-processing technique for binary classification: rather than
wrapping another estimator or adjusting an existing model's predictions, it
fits a logistic regression whose training objective penalizes predictions that
depend on the sensitive feature.
The penalty is the *prejudice index*, a sample estimate of the mutual
information between the predictions and the sensitive feature, so a larger
penalty means the sensitive feature explains less of the prediction.

Input and output
----------------

The input consists of features :math:`\mathbf{X},` binary labels :math:`y,`
and a categorical sensitive feature :math:`S` with a finite set of values.
The estimator learns one weight vector :math:`\mathbf{w}_s` per value :math:`s`
of :math:`S`, so a group is one value of the sensitive feature.
Because the group selects the weights, the sensitive feature has to be passed
to :meth:`~PrejudiceRemover.fit`, :meth:`~PrejudiceRemover.predict` and
:meth:`~PrejudiceRemover.predict_proba`.
This is the same calling convention as
:class:`~fairlearn.postprocessing.ThresholdOptimizer`.

The objective follows equation (12) of the paper:

.. math::

    - \sum_{(y_i, \mathbf{x}_i, s_i) \in D} \ln M[y_i \mid \mathbf{x}_i, s_i; \Theta]
    + \eta R_{PR}(D, \Theta)
    + \frac{1}{2C} \sum_s \lVert \mathbf{w}_s \rVert_2^2,

where :math:`\Theta` collects the weight vectors, :math:`\eta` is
:code:`eta`, :code:`C` is the inverse strength of the L2 penalty, and the
prejudice remover regularizer is

.. math::

    R_{PR}(D, \Theta) = \sum_{(\mathbf{x}_i, s_i) \in D} \sum_{y \in \{0, 1\}}
    M[y \mid \mathbf{x}_i, s_i; \Theta]
    \ln \frac{\Pr[\hat{Y} = y \mid s_i]}{\Pr[\hat{Y} = y]}.

:math:`\Pr[\hat{Y} \mid s]` and :math:`\Pr[\hat{Y}]` are estimated from the
model's own predictions on the training set, as in equations (9) and (10) of
the paper.
The regularizer is therefore zero exactly when the predictions are independent
of the sensitive feature, and it grows as the two become more dependent.

Trade-off
---------

:code:`eta` controls the trade-off:

* with :code:`eta=0` the regularizer is switched off. The fit then reduces to
  one logistic regression per group, i.e. one
  :class:`sklearn.linear_model.LogisticRegression` with the same :code:`C`
  fitted on each group separately.
* larger values of :code:`eta` lower the prejudice index, at the cost of
  predictive accuracy. Values between :code:`1` and :code:`100` are a
  reasonable range to explore.

The regularizer is estimated on the training data, so what it drives to zero is
the dependence between the *training* predictions and the sensitive feature. The
effect on held-out data depends on the data and on the size of :code:`eta`, and
:code:`eta` is best chosen with a validation split.

Example
-------

The example below fits the model twice, once without the regularizer and once
with it, and reports the demographic parity difference of each on the same
data.

>>> import numpy as np
>>> from fairlearn.linear_model import PrejudiceRemover
>>> from fairlearn.metrics import demographic_parity_difference
>>> rng = np.random.default_rng(0)
>>> n = 500
>>> sensitive = rng.integers(0, 2, size=n)
>>> X = rng.normal(size=(n, 2))
>>> y = (X[:, 0] + 3 * sensitive + rng.normal(size=n) > 1).astype(int)
>>> small = PrejudiceRemover(eta=0.0).fit(X, y, sensitive_features=sensitive)
>>> large = PrejudiceRemover(eta=100.0).fit(X, y, sensitive_features=sensitive)
>>> without_regularizer = demographic_parity_difference(
...     y, small.predict(X, sensitive_features=sensitive), sensitive_features=sensitive
... )
>>> with_regularizer = demographic_parity_difference(
...     y, large.predict(X, sensitive_features=sensitive), sensitive_features=sensitive
... )
>>> bool(with_regularizer < without_regularizer)
True

The fitted model exposes the regularizer's value as
:attr:`~PrejudiceRemover.prejudice_index_` and the per-group coefficients as
:attr:`~PrejudiceRemover.coef_`.

Limitations
-----------

* Only binary classification and a single categorical sensitive feature are
  supported. A continuous sensitive feature has no groups to learn weights for,
  and several sensitive features would define groups by their combination,
  which is not what the paper's regularizer estimates.
* The estimator is not interchangeable with a
  :class:`sklearn.linear_model.LogisticRegression` in a
  :class:`~sklearn.pipeline.Pipeline`, because :code:`predict` needs the
  sensitive feature. Use it where
  :class:`~fairlearn.postprocessing.ThresholdOptimizer` would be used.
* The objective is non-convex in the weights, so the result depends on the
  starting point. The optimization starts from zero, which makes repeated fits
  on the same data give the same result.
