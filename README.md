# Reductions for Fair Machine Learning

A python package that implements the black-box approach to fair classification described in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453).

## Installation

The package can be installed via `pip install fairlearn`. To verify that it works, download `test_fairlearn.py` from the repository and run `python test_fairlearn.py`.

Instead of installing the package, you can clone the repository locally via `git clone git@github.com:Microsoft/fairlearn.git`. To verify that the package works, run `python test_fairlearn.py` in the root of the repository.

## Usage

The function `expgrad` in the module `fairlearn.classred` implements the reduction of fair classification to weighted binary classification. Any learner that supports weighted binary classification can be provided as input for this reduction. Two common fairness definitions are provided in the module `fairlearn.moments`: demographic parity (class `DP`) and equalized odds (class `EO`). See the file `test_fairlearn.py` for example usage of `expgrad`.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Maintainers

fairlearn is maintained by:

- **@MiroDudik**

### Releasing

If you are the current maintainer of this project:

1. Create a branch for the release: `git checkout -b release-vxx.xx`
1. Ensure that all tests return "ok": `python test_fairlearn.py`
1. Bump the module version in `fairlearn/__init__.py`
1. Make a pull request to Microsoft/fairlearn
1. Merge Microsoft/fairlearn pull request
1. Tag and push: `git tag vxx.xx; git push --tags`

## Reporting Security Issues

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).
