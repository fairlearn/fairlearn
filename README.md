[![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Nightly?branchName=master)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=13&branchName=master) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![pypi badge](https://img.shields.io/badge/pypi-0.2.0-blue)



# Fairness in machine learning - Mitigation algorithms

A Python package that implements a variety of algorithms that mitigate unfairness in supervised machine learning.

## Current Release

* The current stable release is available at [FairLearn v0.2.0](https://github.com/microsoft/fairlearn/tree/v0.2.0).

* Our latest work differs substantively from version 0.2.0, please visit the repo for further information. In particular, look at [Existing users: How to onboard to fairlearn v0.3+](#existing).

#  Algorithms

A variety of fairness-related algorithms to mitigate bias are included:

| algorithm | description | classification/regression | protected attributes | supported fairness definitions |
| --- | --- | --- | --- | --- |
| `fairlearn.` `reductions.` `ExponentiatedGradient` | Black-box approach to fair classification described in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | categorical | DP, EO |
| `fairlearn.` `reductions.` `GridSearch` | Black-box approach described in Section 3.4 of the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | binary | DP |
| `fairlearn.` `reductions.` `GridSearch` | Grid search for regression | regression | binary | BGL |
| `fairlearn.` `post_processing.` `ThresholdOptimizer` | Post-processing algorithm based on the paper [Equality of Opportunity in Supervised Learning](https://arxiv.org/pdf/1610.02413.pdf)| binary classification | categorical | DP, EO |

DP refers to Demographic Parity, EO to Equalized Odds, and BGL to Bounded Group Loss. For more information on these and other terms we use in this repository please refer to the [terminology page](TERMINOLOGY.md).

To request additional algorithms or fairness definitions, please open a new issue.

# <a name="existing"></a>

# Existing users: How to onboard to fairlearn v0.3+

<details>
<summary>
<strong>
<em>
Onboarding guide
</em>
</strong>
</summary>

As of version 0.2 fairlearn contained only the exponentiated gradient method. The fairlearn repository now has a more comprehensive scope and aims to incorporate other methods as specified above. The same exponentiated gradient technique is now located under `fairlearn.reductions.ExponentiatedGradient` as a class. While in the past one could have run

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
exponentiated_gradient = ExponentiatedGradient(estimator, constraints=DemographicParity())
exponentiated_gradient.fit(X, y, group_data)
randomized_predictions = exponentiated_gradient.predict(X)
```

Please open a new issue if you encounter any problems.

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

To contribute please check our [Contributing guide](CONTRIBUTING.md).

# Maintainers

fairlearn is maintained by:

- **@MiroDudik**
- **@romanlutz**
- **@riedgar-ms**
- **@bethz**

## Releasing

If you are the current maintainer of this project:

1. Create a branch for the release: `git checkout -b release-vxx.xx`
1. Ensure that all tests return "ok": `python -m pytest -s ./test`
1. Bump the module version in `fairlearn/__init__.py`
1. Make a pull request to Microsoft/fairlearn
1. Merge Microsoft/fairlearn pull request
1. Tag and push: `git tag vxx.xx; git push --tags`

# Issues

## Regular (non-security) issues
Please submit a report through [GitHub issues](https://github.com/microsoft/fairlearn/issues). A maintainer will respond promptly as follows:
- bug: triage as `bug` and provide estimated timeline based on severity
- feature request: triage as `feature request` and provide estimated timeline
- question or discussion: triage as `question` and respond or notify/identify a suitable expert to respond

Maintainers are supposed to link duplicate issues when possible.


## Reporting security issues

Please take a look at our guidelines for reporting [security issues](SECURITY.md).
