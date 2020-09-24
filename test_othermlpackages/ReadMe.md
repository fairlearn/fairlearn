# Tests for other Machine Learning Packages

The main test suite for Fairlearn uses estimators from SciKit-Learn.
However, we wish to ensure that Fairlearn works with other machine
learning packages, provided they implement the SciKit-Learn Estimator
interface (with `sample_weight=` on the `fit()` method).
This directory contains tests which demonstrate Fairlearn working with
 various other packages.

We have made this a separate directory because machine learning
packages tend to be heavyweight, and different packages may have
conflicting dependencies.
To avoid making our developer install too large (at best) or impossible
to satisfy (at worst), we separate these tests out from our main test
suite.
We have separate directories for other machine learning packages (`lightgbm`,
`xgboost` etc.) and each one has a `conda-env.yaml` file, which 
the required dependencies.

To run the tests for a particular package, do the following from the
repository root:
1. Create a new conda environment with
`conda env create -n <env-name> -f ./test_othermlpackages/<package>/conda-env.yaml`
1. Activate the environment `conda activate <env-name>`
1. Install Fairlearn with `pip install .`
1. Run the tests with `python -m pytest ./test_othermlpackages/`

Note that the tests are very much smoke tests.
They ensure that the algorithms run without trouble, but they do not check the
results in any sort of detail.