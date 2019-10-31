# Contributing to fairlearn

This project welcomes contributions and suggestions.

## Contributor license agreement
Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

## Code of conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development process
Development happens against the `master` branch following the [GitHub flow model](https://guides.github.com/introduction/flow/). Contributors create feature branches off of `master`, and their pull requests should target the `master` branch. Maintainers will review pull requests within two business days.

Pull requests against `master` trigger automated pipelines that are run through Azure DevOps. Additional test suites are run periodically. When adding new code paths or features tests are a requirement to complete a pull request. They should be added in the `test` directory.

### Investigating automated test failures
For every pull request to `master` with automated tests you can check the logs of the tests to find the root cause of failures. Our tests currently run through Azure Pipelines with steps for setup, test suites, and teardown. The `Checks` view of a pull request contains a link to the [Azure Pipelines Page](dev.azure.com/responsibleai/fairlearn/_build/results). All the steps are represented in the Azure Pipelines page, and you can see logs by clicking on a specific step. If you encounter problems with this workflow please reach out through the `Issues`.

## API
<div id="api">

This section heavily relies on the definitions from our [terminology guide](TERMINOLOGY.md). Specifically, we use the terms "reduction", "sensitive features", "moment", and "parity" in the following.

For all disparity mitigation methods algorithm-specific parameters are passed to the constructor. The methods to fit a mitigator and predict values with the resulting model shall resemble the APIs used by scikit-learn as much as possible for the sake of ease of use. Any deviations are noted below.

### Reductions

Reductions require an estimator to be passed that implements the `fit` method with the `sample_weight` argument. The constraints for reductions are all moments (see `fairlearn.reductions`) passed as instances of classes inheriting from `Moment`. Moments are vector functions that we use to formalize our constraints. The moment objects need to be passed as constraints to the constructor of a reduction technique, which internally uses the constraints to solve the optimization problem.

```python
constraints = Moment()
reduction = Reduction(estimator, objective=objective, constraints=constraints, **kwargs)
```

Reduction-based disparity mitigation algorithms (such as the ones under `fairlearn.reductions`) provide `fit`, `predict`, and `_pmf_predict` methods with the following signatures:

```python
reduction.fit(X, Y, sensitive_features=sensitive_features)
reduction.predict(X)
reduction._pmf_predict(X)
```

where `sensitive_features` contains data on which group a sample belongs to. As of now, sensitive features can only be provided through `sensitive_features`. In the future we plan to allow specifying specific columns of `X` as sensitive features, in which case `sensitive_features` would be optional.

### Post-processing methods

Post-processing methods require an already trained predictor. For consistency we also provide the option to pass an estimator instead, and will call `fit` internally. For post-processing methods we provide the `constraints` argument in the form of a string.

```python
post_processor = PostProcessing(unconstrained_predictor=predictor, constraints=constraints, **kwargs)
post_processor = PostProcessing(estimator=estimator, constraints=constraints, **kwargs)
```

Post-processing methods (such as the ones under `fairlearn.post_processing`) also provide the same functions as the reductions above albeit with `sensitive_features` as a required argument for `predict` and `_pmf_predict`. In the future we will make `sensitive_features` optional if the sensitive features are already provided through `X`.

```python
post_processor.fit(X, Y, sensitive_features=sensitive_features)
post_processor.predict(X, sensitive_features=sensitive_features)
post_processor._pmf_predict(X, sensitive_features=sensitive_features)
```
</div>


# Creating new releases

If you are one of the current maintainers of this project, follow this checklist to create a new release:

1. Ensure that all builds run successfully on all operating systems and python versions on that branch
1. Bump the module version in `fairlearn/__init__.py`
1. Make a pull request to fairlearn/fairlearn
1. Merge fairlearn/fairlearn pull request
1. Tag and push: `git tag vxx.xx; git push --tags`
1. Remove old build files: `git clean -xdf`
1. Upload new package version to pypi
    ```python
    python setup.py sdist bdist_wheel
    python -m twine upload  dist/* --verbose
    ```
