[![Build Status](https://img.shields.io/azure-devops/build/responsibleai/fairlearn/6/dev?failed_label=bad&passed_label=good&label=GatedCheckin%3ADev)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=1&branchName=dev) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![pypi badge](https://img.shields.io/badge/pypi-0.2.0-blue)

# Fairness in Machine Learning - Mitigation algorithms

A Python package that implements a variety of fairness-related algorithms to mitigate bias including:

| algorithm | description | classification/regression | protected attributes | supported fairness definitions |
| --- | --- | --- | --- | --- |
| `fairlearn.` `expgrad` | Black-box approach to fair classification described in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | categorical | DP, EO |
| `fairlearn.` `reductions.` `GridSearch` | As described in Section 3.4 in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | binary | DP |
| `fairlearn.` `reductions.` `GridSearch` | Grid Search for regression | regression | binary | BGL |
| `fairlearn.` `post_processing.` `ROCCurveBasedPostProcessing` | Post-processing algorithm based on the paper [Equality of Opportunity in Supervised Learning](https://arxiv.org/pdf/1610.02413.pdf)| binary classification | categorical | DP, EO |

DP refers to Demographic Parity, EO to Equalized Odds, and BGL to Bounded Group Loss. For more information on these and other terms we use in this repository please refer to [the Terminology page](Terminology.md).

To request additional algorithms or fairness definitions, please open a new issue.

# Existing users: How to onboard to fairlearn v0.3+

<details>
<summary>
<strong>
<em>
Onboarding guide
</em>
</strong>
</summary>

As of version 0.2 fairlearn contained only the exponentiated gradient method. The same method is now located under `fairlearn.reductions.ExponentiatedGradient`. While in the past one could have run

```python
import numpy as np
from fairlearn.classred import expgrad
from fairlearn.moments import DP

estimator = LogisticRegression()  # or any other estimator
exponentiated_gradient_result = expgrad(X, group_data, y, estimator, constraints=DP())
positive_probabilities = exponentiated_gradient_result.best_classifier(X)
randomized_predictions = (positive_probabilities >= np.random.rand(len(positive_probabilities))) * 1
```

the equivalent operation is now

```python
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions.moments import DemographicParity

estimator = LogisticRegression()  # or any other estimator
constraints = DemographicParity()
exponentiated_gradient = ExponentiatedGradient(estimator, constraints=constraints)
exponentiated_gradient.fit(X, y, group_data)
randomized_predictions = exponentiated_gradient.predict(X)
```

</details>

# Installation

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

# Usage

For common usage refer to the [Jupyter notebooks](./notebooks) and our [API guide](CONTRIBUTING.md#api)

# Contributing

To contribute please check our [Contributing Guide](CONTRIBUTING.md).

# Maintainers

fairlearn is maintained by:

- **@MiroDudik**
- **@romanlutz**
- **@riedgar-ms**

## Releasing

If you are the current maintainer of this project:

1. Create a branch for the release: `git checkout -b release-vxx.xx`
1. Ensure that all tests return "ok": `python -m pytest -s ./test`
1. Bump the module version in `fairlearn/__init__.py`
1. Make a pull request to Microsoft/fairlearn
1. Merge Microsoft/fairlearn pull request
1. Tag and push: `git tag vxx.xx; git push --tags`

# Issues

## Regular (non-Security) Issues
Please submit a report through [Github issues](https://github.com/microsoft/fairlearn/issues). A maintainer will respond within 24 hours to handle the issue as follows:
- bug: triage as `bug` and provide estimated timeline based on severity
- feature request: triage as `feature request` and provide estimated timeline
- question or discussion: triage as `question` and respond or notify/identify a suitable expert to respond

Maintainers are supposed to link duplicate issues when possible.


## Reporting Security Issues

Please take a look at our guidelines for reporting [security issues](SECURITY.md).
