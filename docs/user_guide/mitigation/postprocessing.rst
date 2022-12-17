.. _postprocessing:

Postprocessing
--------------

.. currentmodule:: fairlearn.postprocessing

Fairlearn currently supports one postprocessing technique,
:class:`ThresholdOptimizer`.

.. _threshold_optimizer:

Threshold optimizer
^^^^^^^^^^^^^^^^^^^

:class:`ThresholdOptimizer` is based on the paper
*Equality of Opportunity in Supervised Learning*
:footcite:`hardt2016equality`.
Unlike other mitigation techniques :class:`ThresholdOptimizer` is built to
satisfy the specified fairness criteria exactly and with no remaining
disparity.
In many cases this comes at the expense of performance, for example, with
significantly lower accuracy.
Regardless, it is a useful data point to compare results with.
Importantly, :class:`ThresholdOptimizer` requires the sensitive features to be
available at deployment time (i.e., for the :code:`predict` method).

For each sensitive feature value, :class:`ThresholdOptimizer` creates separate
thresholds and applies them to the predictions of the user-provided
:code:`estimator`.
To decide on the thresholds it generates all possible thresholds and selects
the best combination in terms of the :code:`objective` and the fairness
:code:`constraints`.
This process is visualized below using
:func:`fairlearn.postprocessing.plot_threshold_optimizer`.
To ensure that the fairness constraint is upheld :class:`ThresholdOptimizer`
simultaneously walks along all the curves while monitoring the objective
value, and picks the spot on the x-axis with the optimal objective weighed by
the sizes of the groups corresponding to each sensitive feature value.
The selected point is marked by the dashed blue line.
The intersections with the curves are represented through interpolation of the
thresholds we used to create the plot.
To achieve an exact match between the sensitive feature groups in terms of the
constraint, one typically needs to randomize between two thresholds.

To illustrate its behavior, let's examine what this looks like with
demographic parity as the fairness constraint.

.. doctest:: mitigation_postprocessing
    :options:  +NORMALIZE_WHITESPACE

    >>> import json
    >>> import pandas as pd
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.compose import make_column_selector as selector
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>> from fairlearn.datasets import fetch_adult
    >>> from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
    >>> from fairlearn.reductions import DemographicParity, ExponentiatedGradient
    >>> data = fetch_adult(as_frame=True)
    >>> X_raw = data.data
    >>> y = (data.target == ">50K") * 1
    >>> A = X_raw["sex"]
    >>> (X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
    ...     X_raw, y, A, test_size=0.3, random_state=12345, stratify=y)
    >>> X_train = X_train.reset_index(drop=True)
    >>> X_test = X_test.reset_index(drop=True)
    >>> y_train = y_train.reset_index(drop=True)
    >>> y_test = y_test.reset_index(drop=True)
    >>> A_train = A_train.reset_index(drop=True)
    >>> A_test = A_test.reset_index(drop=True)
    >>> numeric_transformer = Pipeline(
    ...     steps=[
    ...         ("impute", SimpleImputer()),
    ...         ("scaler", StandardScaler()),
    ...     ]
    ... )
    >>> categorical_transformer = Pipeline(
    ...     [
    ...         ("impute", SimpleImputer(strategy="most_frequent")),
    ...         ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ...     ]
    ... )
    >>> preprocessor = ColumnTransformer(
    ...     transformers=[
    ...         ("num", numeric_transformer, selector(dtype_exclude="category")),
    ...         ("cat", categorical_transformer, selector(dtype_include="category")),
    ...     ]
    ... )
    >>> pipeline = Pipeline(
    ...     steps=[
    ...         ("preprocessor", preprocessor),
    ...         (
    ...             "classifier",
    ...             LogisticRegression(solver="liblinear", fit_intercept=True),
    ...         ),
    ...     ]
    ... )
    >>> threshold_optimizer = ThresholdOptimizer(
    ...     estimator=pipeline,
    ...     constraints="demographic_parity",
    ...     predict_method="predict_proba",
    ...     prefit=False,
    ... )
    >>> threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)
    ThresholdOptimizer(estimator=Pipeline(steps=[('preprocessor',
                                                  ColumnTransformer(transformers=[('num',
                                                                                   Pipeline(steps=[('impute',
                                                                                                    SimpleImputer()),
                                                                                                   ('scaler',
                                                                                                    StandardScaler())]),
                                                                                   <sklearn.compose._column_transformer.make_column_selector object at 0x...>),
                                                                                  ('cat',
                                                                                   Pipeline(steps=[('impute',
                                                                                                    SimpleImputer(strategy='most_frequent')),
                                                                                                   ('ohe',
                                                                                                    OneHotEncoder(handle_unknown='ignore'))]),
                                                                                   <sklearn.compose._column_transformer.make_column_selector object at 0x...>)])),
                                                 ('classifier',
                                                  LogisticRegression(solver='liblinear'))]),
                       predict_method='predict_proba')
    >>> threshold_optimizer.predict(X_test, sensitive_features=A_test)
    array([0, 0, 1, ..., 0, 0, 1])
    >>> threshold_rules_by_group = threshold_optimizer.interpolated_thresholder_.interpolation_dict
    >>> print(json.dumps(threshold_rules_by_group, default=str, indent=4))
    {
        "Female": {
            "p0": 0.8004605263157903,
            "operation0": "[>0.20326115065158867]",
            "p1": 0.19953947368420966,
            "operation1": "[>0.18733195463637906]"
        },
        "Male": {
            "p0": 0.03672549019607803,
            "operation0": "[>0.6820004991747526]",
            "p1": 0.9632745098039219,
            "operation1": "[>0.6662093209959495]"
        }
    }
    >>> plot_threshold_optimizer(threshold_optimizer)

When calling :code:`predict` :class:`ThresholdOptimizer` uses one of the
thresholds at random based on the probabilities :math:`p_0` and :math:`p_1`.
The results can be interpreted as follows based on the following formula for
the probability to predict label 1:

.. math::

    p_0 \cdot \text{operation}_0(\text{score}) + p_1 \cdot \text{operation}_1(\text{score})


- "Male": :math:`0.99989 \cdot \mathbb{I}(\text{score}>0.5) + 0.00011 \cdot \mathbb{I}(\text{score}>-\infty)`

  - if the score is less or equal to :math:`0.5` predict 1 with probability
    :math:`0.00011`
  - if the score is above :math:`0.5` predict 1

- "Female: :math:`0.95272 \cdot \mathbb{I}(\text{score}>0.5) + 0.04728 \cdot \mathbb{I}(\text{score}>-\infty)`

  - if the score is less or equal to :math:`0.5` predict 1 with probability
    :math:`0.04728`
  - if the score is above :math:`0.5` predict 1

The thresholds are actually the same for both groups here, which
does not always have to be true. As a consequence, the only difference is
the probability to predict label 1 in the case where the score is below
:math:`0.5`.

.. note::

    The :code:`flip` argument on the constructor indicates whether flipped
    thresholds can be used. Flipped constraints such as `"<0.6"`
    indicate that the underlying estimator's scores do not exhibit the
    expected monotonicity and any such instance should be inspected.

.. note::

    :class:`ThresholdOptimizer` expects an estimator that provides it with
    scores. While the output of :class:`ThresholdOptimizer` is binary, the
    input need not be. In fact, real valued input, e.g. from a regressor,
    provides it with many more options to create thresholds. For :math:`n`
    input data points with :math:`m \leq n` different score values it has
    :math:`m+1` different thresholds. At each threshold one can create one of
    two thresholding rules, i.e. functions that indicate which data points
    get label :code:`1` based on their score.

The following combinations of fairness criteria and objectives are available:

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   *  - fairness criteria
      - objectives
   *  - :code:`equalized_odds`
      - :code:`accuracy_score`,
        :code:`balanced_accuracy_score`
   *  - :code:`demographic_parity`,
        :code:`false_positive_rate_parity`,
        :code:`false_negative_rate_parity`,
        :code:`true_positive_rate_parity`,
        :code:`true_negative_rate_parity`
      - :code:`selection_rate`,
        :code:`true_positive_rate`,
        :code:`true_negative_rate`,
        :code:`accuracy_score`,
        :code:`balanced_accuracy_score`


.. _threshold_optimizer_equalized_odds:

`ThresholdOptimizer` for equalized odds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For equalized odds the behavior is somewhat different because it requires two
metrics to be equal across all groups (true and false positive rate).
The code below visualizes the threshold selection process with an ROC curve.
The ROC curves consist of the true and false positive rates for each
of the thresholding rules, with separate ones per sensitive feature value.
Note that the plot omits points that are within the convex hull of points.

.. _printed_thresholds:

.. doctest:: mitigation_postprocessing
    :options:  +NORMALIZE_WHITESPACE

    >>> data = fetch_adult(as_frame=True)
    >>> logistic_regression = LogisticRegression()
    >>> threshold_optimizer = ThresholdOptimizer(
    ...     estimator=logistic_regression,
    ...     constraints="equalized_odds",
    ...     objective="accuracy_score")
    >>> threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)
    ThresholdOptimizer(constraints='equalized_odds', estimator=LogisticRegression())
    >>> threshold_rules_by_group = threshold_optimizer.interpolated_thresholder_.interpolation_dict
    >>> print(json.dumps(
    ...     threshold_rules_by_group,
    ...     default=str,
    ...     indent=4))
    {
        "Female": {
            "p_ignore": 0.13186000824177083,
            "prediction_constant": 0.035,
            "p0": 0.9943698649710652,
            "operation0": "[>0.5]",
            "p1": 0.005630135028934835,
            "operation1": "[>-inf]"
        },
        "Male": {
            "p_ignore": 0.0,
            "prediction_constant": 0.035,
            "p0": 0.9999261555292187,
            "operation0": "[>0.5]",
            "p1": 7.384447078129242e-05,
            "operation1": "[>-inf]"
        }
    }
    >>> plot_threshold_optimizer(threshold_optimizer)

The printed thresholding rules influence the predictions, so we need to first
understand the formula that puts all the parts together.
As before, to reach the optimal solution we have to interpolate between
points by creating the linear combination of the two thresholds as
:math:`p_0 \cdot \text{operation}_0(\text{score}) + p_1 \cdot \text{operation}_1(\text{score})`.
The goal in this case with equalized odds is to reach equal true and false
positive rates, which are conveniently plotted on the y- and x-axes,
while optimizing the objective value.
It follows that they'd be equal at any spot where the two curves intersect.
We can relax this a little bit since in many cases one lies strictly above
the other, just like in this example.
Imagine the diagonal of the ROC plot from :math:`(0,0)` to :math:`(1,1)`.
It is usually not considered as useful since we could achieve the same result
with random classifiers with probabilities to get label 1 ranging from 0 to 1.
If we have one curve strictly below the other, we can still get their true and
false positive rates to match by "pulling" the higher one towards the
diagonal. In reality we consider the overlap, so only the points on the lower
curve are available to us. From those points we choose the one with the best
value in terms of the overall objective (not per group).
Stepping below the Pareto curve for any sensitive feature group means
introducing a probability (:code:`p_ignore`) that we need to toss a biased
coin whose bias is indicated by :code:`prediction_constant`.

Mathematically, the following formula represents this interpretation of the
chart, where :math:`c` represents the `prediction_constant` from the printed
dictionary above.

.. math::

    p_{\text{ignore}} \cdot c + (1-p_{\text{ignore}}) \cdot \left(p_0 \cdot \text{operation}_0(\text{score}) + p_1 \cdot \text{operation}_1(\text{score})\right)

In other words, we create a linear combination of the binary thresholding rule
outputs. :math:`p_0` and :math:`p_1` indicate how close we are to the points
representing the two thresholding rules, and :math:`p_0+p_1=1`.
:math:`p_{\text{ignore}}` is :math:`0` if the curve does not need to be "pulled"
towards the diagonal, i.e., when the selected solution lies on the curve
itself as opposed to below.
Above, the curve for "Male" is strictly below the curve for "Female", so
:math:`p_{\text{ignore}}=0` for "Male" and non-zero for "Female".

Importantly, :math:`p_{\text{ignore}}` is only required for equalized odds.
We want to stress on the fact that it is crucial to understand the
implications of such a result.
Predictions of the resulting :class:`ThresholdOptimizer` model are determined
randomly based on the probabilities to predict label 1 that the formula
produces.
In this particular example :code:`operation1` will always be true since every
score is larger than negative infinity.
For "Male" that means

.. math::

    0.99993 \cdot \mathbb{I}(\text{score} > 0.5) + 0.00007 \cdot \mathbb{I}(\text{score} > \infty)

- predict 1 with probability :math:`1-0.999926=0.000074` if the score is less
  or equal to :math:`0.5`. This means that there is a very low probability
  (less than a hundredth of a percent) of getting label 1 regardless of the
  features.
- predict 1 if the score is greater than :math:`0.5`.

For "Female" the result includes the :math:`p_{\text{ignore}}` term, and one
of the thresholds is set to always be true:

.. math::

    0.13186 \cdot 0.035 + (1-0.13186) \cdot \left(0.99437 \cdot \mathbb{I}(\text{score}>0.5) + 0.00563 \cdot \mathbb{I}(\text{score}>\infty)\right)


- predict 1 with probability
  :math:`0.13186 \cdot 0.035 + (1-0.13186) \cdot 0.00563 = 0.0095` if the
  score is less or equal to :math:`0.5`. This represents a 1% chance of
  getting label 1 regardless of the features.
- predict 1 if the score is greater than :math:`0.5`

We want to emphasize that a non-zero probability to get label 1 is not
acceptable in some application contexts, so it needs to be carefully evaluated
whether :class:`ThresholdOptimizer` should be used.

Due to the separate thresholding rules per sensitive feature value, one might
argue that this constitutes
`disparate treatment <https://en.wikipedia.org/wiki/Disparate_treatment>`_ in
certain contexts where this is legally relevant.

Using a pre-trained estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, :class:`ThresholdOptimizer` trains the passed estimator using its
:code:`fit()` method. If :code:`prefit` is set to :code:`True`,
:class:`ThresholdOptimizer` does not call :code:`fit()` on the estimator and
assumes that it is already trained.

.. doctest:: mitigation

    >>> prefit_logistic_regression = LogisticRegression()
    >>> prefit_logistic_regression.fit(X_train, y_train)
    LogisticRegression()
    >>> threshold_optimizer = ThresholdOptimizer(
    ...     estimator=prefit_logistic_regression,
    ...     constraints="demographic_parity",
    ...     objective="accuracy_score",
    ...     prefit=True)
    >>> threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)
    ThresholdOptimizer(estimator=LogisticRegression(), prefit=True)

If it detects that the estimator may not be fitted as defined by
`scikit-learn's check <https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html>`_
it prints a warning, but still proceeds with the assumption that the user
knows what they are doing. Not every machine learning package adheres to
scikit-learn's convention of setting all members with trailing underscore
during :code:`fit()`, so this is unfortunately an imperfect check.
