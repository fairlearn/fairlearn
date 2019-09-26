[![Build Status](https://img.shields.io/azure-devops/build/responsibleai/fairlearn/6/dev?failed_label=bad&passed_label=good&label=GatedCheckin%3ADev)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=1&branchName=dev) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![pypi badge](https://img.shields.io/badge/pypi-0.2.0-blue)

# Reductions for Fair Machine Learning

?Thoughts for a new title?


A Python package that implements the black-box approach to fair classification described in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453).


## Try our notebooks

[![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/microsoft/fairlearn)

## Installation

The package can be installed via `pip install fairlearn`. To verify that it works, download `test_fairlearn.py` from the repository and run `python test_fairlearn.py`.

Instead of installing the package, you can clone the repository locally via `git clone git@github.com:Microsoft/fairlearn.git`.  To verify that the package works, run
`python test_fairlearn.py` in the root of the repository.

<details><summary>Step-by-step Instructions for cloning and using the repository</summary>

This repo uses Anaconda to simplify package and environment management.

To setup on your local machine:

<details><summary><strong><em>1. Set up Environment</em></strong></summary>

    a. Install Anaconda with Python >= 3.6 
       [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) is a quick way to get started.

 
    b. Create conda environment named interp and install packages

```
    conda create --name fairlearn python=3.6 anaconda
    
```

    Optional, additional reading:, [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf), [jupyter](https://pypi.org/project/jupyter/) and [nb_conda](https://github.com/Anaconda-Platform/nb_conda)

<details><summary><strong><em>On Windows: c. Activate conda environment</strong></em></summary>

```
    activate fairlearn
```
</details>

<details><summary><strong><em>On Linux:</em> c. Activate conda environment</em></strong></summary>

```
    source activate fairlearn
```
</details>

</details>
 
<details>

<summary><strong><em>2. Clone the interpret-community repository</em></strong></summary>

a. Clone and cd into the repository
```
git clone https://github.com/Microsoft/Fairlearn
```
</details>

<details>
<summary><strong><em>3. Test that the package works</em></strong></summary>

```
python test_fairlearn.py
```
</details>

<details>
<summary><strong><em>3. Set up and run Jupyter Notebook server </em></strong></summary>

Click here to automatically import these notebooks to Auzre:
[![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/microsoft/fairlearn)

If you prefer to run locally:
a. Install and run Jupyter Notebook
```
if needed:
          pip install jupyter
          conda install nb_conda
then:
jupyter notebook
```
</details>
</details>

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

# Build Status

| Build Type | Branch | Status |
| -- | -- | -- |
| Gated Checkin | Dev | [![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Gated%20Checkin?branchName=dev)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=1&branchName=dev)|
| Notebooks | Dev | [![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Notebooks?branchName=dev)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=8&branchName=dev)|

