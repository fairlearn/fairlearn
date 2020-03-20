# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# On MacOS we need to use TKAgg before importing matplotlib.pyplot.
# This used to work on all platforms until matplotlib 3.2.0 broke it on Linux.
# Consider removing the OS-based if after a future release of matplotlib (issue #320).
import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # noqa: E402
import pkg_resources  # noqa: E402
import pytest  # noqa: E402
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer  # noqa: E402
from fairlearn.postprocessing._constants import DEMOGRAPHIC_PARITY, EQUALIZED_ODDS  # noqa: E402

from .conftest import scores_ex, ExamplePredictor, _data_ex1, _data_ex2, _data_ex3  # noqa: E402


PYTEST_MPL_NOT_INSTALLED_MSG = "skipping plotting tests because pytest-mpl is not installed"

"""Right now the baseline plot comparison doesn't succeed consistently on every
platform and is therefore disabled. To generate the baseline plots run the
following command from the root directory of the repository
python -m pytest test/unit/postprocessing/test_plots.py
    --mpl-generate-path=test/unit/postprocessing/baseline
Make sure to have `pytest-mpl` installed or this will not work.
pytest can run the tests either to check that there are no exceptions (using
a typical pytest command without extra options) or to actually compare the
generated images with the baseline plots (using pytest --mpl)."""


def _fit_and_plot(constraints, plotting_data):
    adjusted_predictor = ThresholdOptimizer(estimator=ExamplePredictor(scores_ex),
                                            constraints=constraints)
    adjusted_predictor.fit(plotting_data.X, plotting_data.y,
                           sensitive_features=plotting_data.sensitive_features)
    fig, (ax) = plt.subplots(1, 1)
    plot_threshold_optimizer(adjusted_predictor, ax=ax, show_plot=False)
    return fig


def is_mpl_installed():
    try:
        pkg_resources.get_distribution("pytest-mpl")
        pkg_resources.get_distribution("matplotlib")
        return True
    except pkg_resources.DistributionNotFound:
        return False


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
@pytest.mark.mpl_image_compare(filename="equalized_odds_ex1.png")
def test_plot_equalized_odds_ex1():
    return _fit_and_plot(EQUALIZED_ODDS, _data_ex1)


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
@pytest.mark.mpl_image_compare(filename="equalized_odds_ex2.png")
def test_plot_equalized_odds_ex2():
    return _fit_and_plot(EQUALIZED_ODDS, _data_ex2)


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
@pytest.mark.mpl_image_compare(filename="equalized_odds_ex3.png")
def test_plot_equalized_odds_ex3():
    return _fit_and_plot(EQUALIZED_ODDS, _data_ex3)


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
@pytest.mark.mpl_image_compare(filename="demographic_parity_ex1.png")
def test_plot_demographic_parity_ex1():
    return _fit_and_plot(DEMOGRAPHIC_PARITY, _data_ex1)


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
@pytest.mark.mpl_image_compare(filename="demographic_parity_ex2.png")
def test_plot_demographic_parity_ex2():
    return _fit_and_plot(DEMOGRAPHIC_PARITY, _data_ex2)


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
@pytest.mark.mpl_image_compare(filename="demographic_parity_ex3.png")
def test_plot_demographic_parity_ex3():
    return _fit_and_plot(DEMOGRAPHIC_PARITY, _data_ex3)
