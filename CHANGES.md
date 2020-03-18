# Changes

### v0.4.5
* Changes to `ThresholdOptimizer`:
  * Separate plotting for `ThresholdOptimizer` into its own plotting function.
  * `ThresholdOptimizer` now performs validations during `fit`, and not during
    `__init__`. It also stores the fitted given estimator in the `estimator_`
    attribute.
  * `ThresholdOptmizer` is now a scikit-learn meta-estimator, and accepts
    an estimator through the `estimator` parameter. To use a pre-fitted
    estimator, pass `prefit=True`.
* Rename arguments of `create_group_metric_set()` to match the dashboard
* Remove `Reduction` base class for reductions methods and replace it with
  `sklearn.base.BaseEstimator` and `sklearn.base.MetaEstimatorMixin`.
* Fix regression in input validation that dropped metadata from `X` if it is
  provided as a `pandas.DataFrame`.

### v0.4.4
* Remove `GroupMetricSet` in favour of a `create_group_metric_set` method
* Add basic support for multiple sensitive features
* Refactor `ThresholdOptimizer` to use mixins from scikit-learn
* Adjust `scipy`, `scikit-learn`, and `matplotlib` requirements to support python 3.8

### v0.4.3

* Various tweaks to `GroupMetricResult` and `GroupMetricSet` for AzureML integration

### v0.4.2

* If methods such as `predict` are called before `fit`, `sklearn`'s
  `NotFittedError` is raised instead of `NotFittedException`, and the latter
  is now removed.

### v0.4.2, 2020-01-24
* Separated out matplotlib dependency into an extension that can be installed via `pip install fairlearn[customplots]`.
* Added a `GroupMetricSet` class to hold collections of `GroupMetricResult` objects

### v0.4.1, 2020-01-09
* Fix to determine whether operating as binary classifier or regressor in dashboard

### v0.4.0, 2019-12-05
* Initial release of fairlearn dashboard

### v0.3.0, 2019-11-01

* Major changes to the API. In particular the `expgrad` function is now implemented by the `ExponentiatedGradient` class. Please refer to the [ReadMe](readme.md) file for information on how to upgrade

* Added new algorithms
  * Threshold Optimization
  * Grid Search
  
* Added grouped metrics


### v0.2.0, 2018-06-20

* registered the project at [PyPI](https://pypi.org/)

* changed how the fairness constraints are initialized (in `fairlearn.moments`), and how they are passed to the fair learning reduction `fairlearn.classred.expgrad`

### v0.1, 2018-05-14

* initial release
