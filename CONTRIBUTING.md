# Contributing to FairLearn

This project welcomes contributions and suggestions.

## Contributor License Agreement
Contributions require you to sign a Developer Certificate of Origin (DCO) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://developercertificate.org/.

When you submit a pull request, a DCO-bot will automatically determine whether you need to provide a DCO and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our DCO.

## Code of Conduct
This project has adopted the [GitHub Community Guidelines](https://help.github.com/en/github/site-policy/github-community-guidelines).

## Development Process
Development happens against the `master` branch following the [GitHub flow model](https://guides.github.com/introduction/flow/). Contributors create feature branches off of `master`, and their pull requests should target the `master` branch. Maintainers will review pull requests within two business days.

Pull requests against `master` trigger automated pipelines that are run through Azure DevOps. Additional test suites are run periodically. When adding new code paths or features tests are a requirement to complete a pull request. They should be added in the `test` directory.

### Investigating automated test failures
For every pull request to `master` with automated tests you can check the logs of the tests to find the root cause of failures. Our tests currently run through Azure Pipelines with steps for setup, test suites, and teardown. The `Checks` view of a pull request contains a link to the [Azure Pipelines Page](dev.azure.com/responsibleai/fairlearn/_build/results). All the steps are represented in the Azure Pipelines page, and you can see logs by clicking on a specific step. If you encounter problems with this workflow please reach out through the `Issues`.

## API
<div id="api">

This section heavily relies on the definitions from our [terminology guide](TERMINOLOGY.md). Specifically, we use the terms "reduction", "group data", "moment", and "parity" in the following.

For all disparity mitigation methods algorithm-specific parameters are passed to the constructor. The methods to fit a mitigator and predict values with the resulting model shall resemble the APIs used by scikit-learn as much as possible for the sake of ease of use. Any deviations are noted below.

### Reductions

Reductions require an estimator to be passed that implements the `fit` method with the `sample_weight` argument. The constraints for reductions are all moments (see `fairlearn.reductions.moments`) passed as instances of classes inheriting from `Moment`. Moments are vector functions that we use to formalize our constraints. The moment objects need to be passed as constraints to the constructor of a reduction technique, which internally uses the constraints to solve the optimization problem.

```python
constraints = Moment()
reduction = Reduction(estimator, objective=objective, constraints=constraints, use_predict_proba=False, **kwargs)
```

Reduction-based disparity mitigation algorithms (such as the ones under `fairlearn.reductions`) provide `fit`, `predict`, and `predict_proba` methods with the following signatures:

```python
reduction.fit(X, Y, group_data)
reduction.predict(X)
reduction.predict_proba(X)
```

where `group_data` contains data on which group a sample belongs to. As of now, grouping data can only be provided through `group_data`. In the future we plan to allow specifying specific columns of `X` as grouping data, in which case `group_data` would be optional.

### Post-processing methods

Post-processing methods require an already trained predictor. For consistency we also provide the option to pass an estimator instead, and will call `fit` internally. For post-processing methods we provide the `constraints` argument in the form of a string.

```python
post_processor = PostProcessing(unconstrained_predictor=predictor, constraints=constraints, **kwargs)
post_processor = PostProcessing(estimator=estimator, constraints=constraints, **kwargs)
```

Post-processing methods (such as the ones under `fairlearn.post_processing`) also provide the same functions as the reductions above albeit with `group_data` as a required argument for `predict` and `_pmf_predict`. In the future we will make `sensitive_features` optional if the grouping data is already provided through `X`.

```python
post_processor.fit(X, Y, sensitive_features=sensitive_features)
post_processor.predict(X, sensitive_features=sensitive_features)
post_processor._pmf_predict(X, sensitive_features=sensitive_features)
```
</div>