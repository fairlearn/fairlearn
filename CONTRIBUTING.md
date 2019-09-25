# Contributing to FairLearn

This project welcomes contributions and suggestions.

## Contributor License Agreement
Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development Process
Development happens against the `master` branch following the [GitHub flow model](https://guides.github.com/introduction/flow/). Contributors create feature branches off of `master`, and their pull requests should target the `master` branch. Maintainers will review pull requests within two business days.

Pull requests against `master` trigger automated tests that are run through Azure DevOps. Additional test suites are run on a rolling basis after check-ins and periodically. When adding new code paths or features consider adding tests in the `test` directory.

### Investigating automated test failures
For every pull request to `master` with automated tests you can check the logs of the tests to find the root cause of failures. Our tests currently run through Azure Pipelines with steps for setup, test suites, and teardown. The `Checks` view of a pull request contains a link to the [Azure Pipelines Page](dev.azure.com/responsibleai/fairlearn/_build/results). All the steps are represented in the Azure Pipelines page, and you can see logs by clicking on a specific step. If you encounter problems with this workflow please reach out through the `Issues`.

## API

Any algorithm-specific parameters are passed to the constructor. Reductions require an estimator to be passed that implements the `fit` method with the `sample_weight` argument. Post-processing algorithms require an already trained predictor. For consistency we also provide the option to pass an estimator instead, and will call `fit` internally.

```python
reduction = Reduction(estimator, constraints=constraints, **kwargs)
post_processor = PostProcessing(fairness_unaware_model=model, disparity_criterion=disparity_criterion, **kwargs)
post_processor = PostProcessing(fairness_unaware_estimator=estimator, disparity_criterion=disparity_criterion, **kwargs)
```

Reduction-based fairness mitigation algorithms (such as the ones under `fairlearn.reductions`) provide `fit`, `predict`, and `predict_proba` methods with the following signatures:

```python
reduction.fit(X, Y, aux_data)
reduction.predict(X)
reduction.predict_proba(X)
```

Post-processing algorithms (such as the ones under `fairlearn.post_processing`) also provide the same functions albeit with `aux_data` as a required argument for `predict` and `predict_proba`

```python
post_processor.fit(X, Y, aux_data)
post_processor.predict(X, aux_data)
post_processor.predict_proba(X, aux_data)
```
