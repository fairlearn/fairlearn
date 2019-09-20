# Fairness in Machine Learning - Mitigation algorithms

A Python package that implements a variety of fairness-related algorithms to mitigate bias including:

| algorithm | description | classification/regression | protected attributes | supported fairness definitions |
| --- | --- | --- | --- | --- |
| `expgrad` | Black-box approach to fair classification described in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | categorical | DP, EO |
| `grid_search.binary_protected_attribute.binary_classification` | As described in Section 3.4 in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | binary | DP, EO |
| `grid_search.binary_protected_attribute.regression` | Grid Search for regression | regression | binary | BGL |
| `post_processing.ROCCurveBasedPostProcessing` | Post-processing algorithm based on the paper [Equality of Opportunity in Supervised Learning](https://arxiv.org/pdf/1610.02413.pdf)| binary classification | categorical | DP, EO |

DP refers to Demographic Parity, EO to Equalized Odds, and BGL to Bounded Group Loss. For more information on these and other terms we use in this repository please refer to [the Terminology page](Terminology.md).

To request additional algorithms or fairness definitions, please open a new issue.

## Installation

The package can be installed via

```python
pip install fairlearn
```

or you can clone the repository locally via

```python
git clone git@github.com:Microsoft/fairlearn.git
```

To verify that it works run

```python
pip install -r requirements.txt
python -m pytest -s ./test
```

## Usage

The function `expgrad` in the module `fairlearn.classred` implements the reduction of fair classification to weighted binary classification. Any learner that supports weighted binary classification can be provided as input for this reduction. Two common fairness definitions are provided in the module `fairlearn.moments`: demographic parity (class `DP`) and equalized odds (class `EO`). See the file `test_fairlearn.py` for example usage of `expgrad`.

## Contributing

This project welcomes contributions and suggestions.

### Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

### Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Development Process
The `master` branch is always in a stable release version. Active development happens on the `dev` branch. Contributors create feature branches off of `dev`, and their pull requests should target the `dev` branch instead of `master`. Maintainers will review pull requests within two business days.

Pull requests against `dev` or `master` trigger automated tests that are run through Azure DevOps. Additional test suites are run on a rolling basis after check-ins and periodically. When adding new code paths or features consider adding tests in the `test` directory.

#### Investigating automated test failures
For every pull request to `dev` or `master` with automated tests you can check the logs of the tests to find the root cause of failures. Our tests currently run through Azure Pipelines with steps for setup, test suites, and teardown. The `Checks` view of a pull request contains a link to the [Azure Pipelines Page](dev.azure.com/responsibleai/fairlearn/_build/results). All the steps are represented in the Azure Pipelines page, and you can see logs by clicking on a specific step. If you encounter problems with this workflow please reach out through the `Issues`.

## Maintainers

fairlearn is maintained by:

- **@MiroDudik**
- **@romanlutz**
- **@riedgar-ms**

### Releasing

If you are the current maintainer of this project:

1. Create a branch for the release: `git checkout -b release-vxx.xx`
1. Ensure that all tests return "ok": `python -m pytest -s ./test`
1. Bump the module version in `fairlearn/__init__.py`
1. Make a pull request to Microsoft/fairlearn
1. Merge Microsoft/fairlearn pull request
1. Tag and push: `git tag vxx.xx; git push --tags`

## Issues

### Regular (non-Security) Issues
Please submit a report through [Github issues](https://github.com/microsoft/fairlearn/issues). A maintainer will respond within 24 hours to handle the issue as follows:
- bug: triage as `bug` and provide estimated timeline based on severity
- feature request: triage as `feature request` and provide estimated timeline
- question or discussion: triage as `question` and respond or notify/identify a suitable expert to respond

Maintainers are supposed to link duplicate issues when possible.


### Reporting Security Issues

Please take a look at our guidelines for reporting [security issues](SECURITY.md).
