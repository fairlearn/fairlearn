# Contributing to fairlearn

This project welcomes contributions and suggestions.

## Developer certificate of origin
Contributions require you to sign a _developer certificate of origin_ (DCO) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://developercertificate.org/.

When you submit a pull request, a DCO-bot will automatically determine whether you need to provide a DCO and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our DCO.

## Code of conduct
This project has adopted the [GitHub community guidelines](https://help.github.com/en/github/site-policy/github-community-guidelines).

## Development process
Development happens against the `master` branch following the [GitHub flow model](https://guides.github.com/introduction/flow/). Contributors create feature branches off of `master`, and their pull requests should target the `master` branch. Maintainers are responsible for prompt review of pull requests.

Pull requests against `master` trigger automated tests that are run through Azure DevOps. Additional test suites are run periodically. When adding new code paths or features, tests are a requirement to complete a pull request. They should be added in the `test` directory.

### Investigating automated test failures
For every pull request to `master` with automated tests, you can check the logs of the tests to find the root cause of failures. Our tests currently run through Azure Pipelines with steps for setup, testing, and teardown. The `Checks` tab of a pull request contains a link to the [Azure Pipelines page](dev.azure.com/responsibleai/fairlearn/_build/results), where you can review the logs by clicking on a specific step in the automated test sequence. If you encounter problems with this workflow, please reach out through [GitHub issues](https://github.com/fairlearn/fairlearn/issues).

## API conventions

This section relies on the definitions from our [terminology guide](TERMINOLOGY.md), including the definitions of "estimator", "reduction", "sensitive features", "moment", and "parity".

### Unfairness mitigation algorithms

Unfairness mitigation algorithms take form of scikit-learn-style estimators. Any algorithm-specific parameters are passed to the constructor. The resulting instance of the algorithm should support methods `fit` and `predict` with APIs resembling those of scikit-learn as much as possible. Any deviations are noted below.

#### Reductions

Reduction constructors require a parameter corresponding to an estimator that implements the `fit` method with the `sample_weight` argument. Parity constraints for reductions are expressed via instances of various subclasses of the class `fairlearn.reductions.Moment`. Formally, instances of the class `Moment` implement vector-valued random variables whose sample averages over the data are required to be bounded (above and/or below).

```python
constraints = Moment()
reduction = Reduction(estimator, constraints)
```

Reductions provide `fit` and `predict` methods with the following signatures:

```python
reduction.fit(X, y, **kwargs)
reduction.predict(X)
```

All of the currently supported parity constraints (subclasses of `Moment`) are based on sensitive features that need to be provided to `fit` as a keyword argument `sensitive_features`. In the future, it will also be possible to provide sensitive features as columns of `X`.

#### Post-processing algorithms

The constructors of post-processing algorithms require an already trained predictor as an argument. As an alternative, it is possible to provide an estimator, and the estimator is then fitted on the data at the beginning of the execution of `fit`. For post-processing algorithms, the `constraints` argument is provided as a string.

```python
postprocessor = PostProcessing(unconstrained_predictor=predictor, constraints=constraints)
postprocessor = PostProcessing(estimator=estimator, constraints=constraints)
```

Post-processing algorithms (such as the ones under `fairlearn.postprocessing`) provide the same functions as the reductions above albeit with `sensitive_features` as a required argument for `predict`. In the future, we will make `sensitive_features` optional if the sensitive features are already provided through `X`.

```python
postprocessor.fit(X, y, sensitive_features=sensitive_features)
postprocessor.predict(X, sensitive_features=sensitive_features)
```


## Creating new releases

If you are one of the current maintainers of this project, follow this checklist to create a new release:

1. Ensure that all builds run successfully on all operating systems and python versions
1. Tag and push the branch point: `git tag vxx.xx-branch; git push --tags`
1. Create a release branch at this tag: `git checkout -b release/vxx.xx`
1. Convert all relative links in `README.md` to direct links to the prospective version tag in GitHub
1. Bump the module version in `fairlearn/__init__.py`
1. Tag and push: `git tag vxx.xx; git push --tags`
1. Verify Nightly builds are Green for the release branch
1. Manually run the notebooks from the release branch
1. Remove old build files: `git clean -xdf`
1. Follow the [PyPI instructions](https://packaging.python.org/tutorials/packaging-projects) to push to **TestPyPI**
1. Verify that package downloads with correct version from TestPyPI and that the project landing page on TestPyPI is correct
1. Use the [PyPI instructions](https://packaging.python.org/tutorials/packaging-projects/) to push to **PyPI** itself
1. In a new Python environment, run `pip install fairlearn`
1. Copy the contents of `test/unit` to temporary directory (outside the cloned github directory) and run `python -m pytest -s test/unit` in the new environment. Check that all tests pass
1. If necessary, update the position of the version tag on the release branch: `git tag vxx.xx; git push --tags --force`

If updates are required from master in the release branch, these should be cherry picked. Be sure to do a *regular merge* for cherry picks, and **not** a squash merge. This will ensure the identity of the updates is maintained between the branches. This also applies if there are bug fixes in the release branch which need to be migrated into master.
