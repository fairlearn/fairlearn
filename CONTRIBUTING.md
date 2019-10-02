# Contributing to FairLearn

This project welcomes contributions and suggestions.

## Contributor License Agreement
Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development Process
Development happens against the `master` branch following the [GitHub flow model](https://guides.github.com/introduction/flow/). Contributors create feature branches off of `master`, and their pull requests should target the `master` branch. Maintainers will review pull requests within two business days.

Pull requests against `master` trigger automated pipelines that are run through Azure DevOps. Additional test suites are run periodically. When adding new code paths or features consider adding tests in the `test` directory.

### Investigating automated test failures
For every pull request to `master` with automated tests you can check the logs of the tests to find the root cause of failures. Our tests currently run through Azure Pipelines with steps for setup, test suites, and teardown. The `Checks` view of a pull request contains a link to the [Azure Pipelines Page](dev.azure.com/responsibleai/fairlearn/_build/results). All the steps are represented in the Azure Pipelines page, and you can see logs by clicking on a specific step. If you encounter problems with this workflow please reach out through the `Issues`.

## API
<div id="api">

This section heavily relies on the definitions from our [terminology guide](TERMINOLOGY.md). Specifically, we use the terms "reduction", "group data", "moment", and "parity" in the following.

For all unfairness mitigation methods algorithm-specific parameters are passed to the constructor. The methods to fit a mitigator and predict values with the resulting model shall resemble the APIs used by scikit-learn as much as possible for the sake of ease of use. Any deviations are noted below.

### Reductions

Reductions require an estimator to be passed that implements the `fit` method with the `sample_weight` argument. The constraints for reductions are all moments (see `fairlearn.reductions.moments`) passed as instances of classes inheriting from `Moment`. Moments are simply vector functions 

```python
reduction = Reduction(estimator, objective=objective, constraints=constraints, use_predict_proba=False, **kwargs)
```

Reduction-based fairness mitigation algorithms (such as the ones under `fairlearn.reductions`) provide `fit`, `predict`, and `predict_proba` methods with the following signatures:

```python
reduction.fit(X, Y, group_data)
reduction.predict(X)
reduction.predict_proba(X)
```

where `group_data` contains data on which group a sample belongs to. As of now, grouping data can only be provided through `group_data`. In the future we plan to allow specifying specific columns of `X` as grouping data, in which case `group_data` would be optional.

### Post-processing methods

Post-processing methods require an already trained predictor. For consistency we also provide the option to pass an estimator instead, and will call `fit` internally. For post-processing methods we provide the `parity_constraints` argument in the form of a string.

```python
post_processor = PostProcessing(unconstrained_model=model, parity_constraints=parity_constraints, **kwargs)
post_processor = PostProcessing(unconstrained_estimator=estimator, parity_constraints=parity_constraints, **kwargs)
```

Post-processing methods (such as the ones under `fairlearn.post_processing`) also provide the same functions as the reductions above albeit with `group_data` as a required argument for `predict` and `predict_proba`. In the future we will make `group_data` optional if the grouping data is already provided through `X`.

```python
post_processor.fit(X, Y, group_data)
post_processor.predict(X, group_data)
post_processor.predict_proba(X, group_data)
```
</div>