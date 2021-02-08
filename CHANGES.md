# Changes
### v0.5.1

### v0.5.0

* Adjust classes to abide by naming conventions for attributes.
* Change `ExponentiatedGradient` signature by renaming argument `T` to
  `max_iter`, `eta_mul` to `eta0`, and by adding `run_linprog_step`.
* API refactoring to separate out different uses of `eps` within
  `ExponentiatedGradient`. It is now solely responsible for setting the L1
  norm bound in the optimization (which controls the excess constraint
  violation beyond what is allowed by the `constraints` object).
  The other usage of `eps` as the right-hand side of constraints is
  now captured directly in the moment classes as follows:
  * Classification moments: `ConditionalSelectionRate` renamed to
    `UtilityParity` and its subclasses have new arguments on the constructor:
    * `difference_bound` - for difference-based constraints such as demographic
      parity difference
    * `ratio_bound_slack` - for ratio-based constraints such as demographic
      parity ratio
    * Additionally, there's a `ratio_bound` argument which represents the
      argument previously called `ratio`.
  * Regression moments: `ConditionalLossMoment` and its subclasses have a new
    argument `upper_bound` with the same purpose for newly enabled regression
    scenarios on `ExponentiatedGradient`.
  For a comprehensive overview of available constraints refer to the new [user
  guide on fairness constraints for reductions methods](https://fairlearn.github.io/user_guide/mitigation.html#reductions).
* Renamed several constraints to create a uniform naming convention according
  to the accepted [metric harmonization proposal](https://github.com/fairlearn/fairlearn-proposals/blob/master/api/METRICS.md):
  * `ErrorRateRatio` renamed to `ErrorRateParity`, and
    `TruePositiveRateDifference` renamed to `TruePositiveRateParity` since the
    desired pattern is `<metric name>Parity` with the exception of
    `EqualizedOdds` and `DemographicParity`.
  * `ConditionalSelectionRate` renamed to `UtilityParity`.
  * `GroupLossMoment` renamed to `BoundedGroupLoss` in order to have a
    descriptive name and for consistency with the paper. Similarly,
    `AverageLossMoment` renamed to `MeanLoss`.
  For a comprehensive overview of available constraints refer to the new [user
  guide on fairness constraints for reductions methods](https://fairlearn.github.io/user_guide/mitigation.html#reductions).
* Added `TrueNegativeRateParity` to provide the opposite constraint of
  `TruePositiveRateParity` to be used with reductions techniques.
* Add new constraints and objectives in `ThresholdOptimizer`
* Add class `InterpolatedThresholder` to represent the fitted `ThresholdOptimizer`
* Add `fairlearn.datasets` module.

### v0.4.6

* Handle case where reductions relabeling results in a single class
* Refactor metrics:
  * Remove `GroupMetricResult` type in favor of a `Bunch`.
  * Rename and slightly update signatures:
    * `metric_by_group` changed to `group_summary`
    * `make_group_metric` changed to `make_metric_group_summary`
  * Add group summary transformers
    `{difference,ratio,group_min,group_max}_from_group_summary`.
  * Add factory `make_derived_metric`.
* Add new metrics:
  * base metrics `{true,false}_{positive,negative}_rate`
  * group summary metrics `<metric>_group_summary`
  * derived metrics `<metric>_{difference,ratio,group_min,group_max}`
  * disparity metrics `{demographic_parity,equalized_odds}_{difference,ratio}`
* Remove metrics:
  * `fallout_rate` in favor of `false_positive_rate`
  * `miss_rate` in favor of `false_negative_rate`
  * `specificity_score` in favor of `true_negative_rate`
* Change from public to private:
  * `mean_{over,under}prediction` and `{balanced_,}root_mean_squared_error`
    changed to the versions with a leading underscore
* Fix warning due to changing default `dtype` when creating an empty
  `pandas.Series`.
* Enable `GridSearch` for more than two sensitive features values.
* Add new disparity constraints for reductions methods as moments in
  `fairlearn.reductions` including:
  * `TruePositiveRateDifference`
  * ratio options for all existing constraints in addition to the default,
    i.e., difference between groups w.r.t. the relevant metric.
* Make `ExponentiatedGradient` require 0-1 labels for classification problems,
  pending a better solution for Issue 339.

### v0.4.5

* Changes to `ThresholdOptimizer`:
  * Separate plotting for `ThresholdOptimizer` into its own plotting function.
  * `ThresholdOptimizer` now performs validations during `fit`, and not during
    `__init__`. It also stores the fitted given estimator in the `estimator_`
    attribute.
  * `ThresholdOptmizer` is now a scikit-learn meta-estimator, and accepts
    an estimator through the `estimator` parameter. To use a pre-fitted
    estimator, pass `prefit=True`.
* Made `_create_group_metric_set_()` private by prepending with `_`.
  Also changed the arguments, so that this routine requires
  dictionaries for the predictions and sensitive features. This is a
  breaking change.
* Remove `Reduction` base class for reductions methods and replace it with
  `sklearn.base.BaseEstimator` and `sklearn.base.MetaEstimatorMixin`.
* Remove `ExponentiatedGradientResult` and `GridSearchResult` in favor of
  storing the values and objects resulting from fitting the meta-estimator
  directly in the `ExponentiatedGradient` and `GridSearch` objects,
  respectively.
* Fix regression in input validation that dropped metadata from `X` if it is
  provided as a `pandas.DataFrame`.

### v0.4.4

* Remove `GroupMetricSet` in favour of a `create_group_metric_set` method
* Add basic support for multiple sensitive features
* Refactor `ThresholdOptimizer` to use mixins from scikit-learn
* Adjust `scipy`, `scikit-learn`, and `matplotlib` requirements to support
  python 3.8

### v0.4.3

* Various tweaks to `GroupMetricResult` and `GroupMetricSet` for AzureML
  integration

### v0.4.2

* If methods such as `predict` are called before `fit`, `sklearn`'s
  `NotFittedError` is raised instead of `NotFittedException`, and the latter
  is now removed.

### v0.4.2, 2020-01-24

* Separated out matplotlib dependency into an extension that can be installed
  via `pip install fairlearn[customplots]`.
* Added a `GroupMetricSet` class to hold collections of `GroupMetricResult`
  objects

### v0.4.1, 2020-01-09

* Fix to determine whether operating as binary classifier or regressor in
  dashboard

### v0.4.0, 2019-12-05

* Initial release of fairlearn dashboard

### v0.3.0, 2019-11-01

* Major changes to the API. In particular the `expgrad` function is now
  implemented by the `ExponentiatedGradient` class. Please refer to the
  [ReadMe](readme.md) file for information on how to upgrade

* Added new algorithms
  * Threshold Optimization
  * Grid Search
  
* Added grouped metrics

### v0.2.0, 2018-06-20

* registered the project at [PyPI](https://pypi.org/)

* changed how the fairness constraints are initialized (in
  `fairlearn.moments`), and how they are passed to the fair learning reduction
  `fairlearn.classred.expgrad`

### v0.1, 2018-05-14

* initial release
