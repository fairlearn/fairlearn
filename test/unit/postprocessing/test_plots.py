# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import pkg_resources
import pytest
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing.plotting import plot_selection_error_curve, plot_roc_curve
from fairlearn.postprocessing._constants import DEMOGRAPHIC_PARITY, EQUALIZED_ODDS

from .conftest import scores_ex, ExamplePredictor, _data_ex1, _data_ex2, _data_ex3


PYTEST_MPL_NOT_INSTALLED_MSG = "skipping plotting tests because pytest-mpl is not installed"


def _fit_and_plot(constraints, plotting_data):
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(scores_ex),
                                            constraints=constraints)
    adjusted_predictor.fit(plotting_data.X, plotting_data.y,
                           sensitive_features=plotting_data.sensitive_features)
    fig, (ax) = plt.subplots(1, 1)
    if constraints == EQUALIZED_ODDS:
        plot_roc_curve(adjusted_predictor, ax=ax, show_plot=False)
    elif constraints == DEMOGRAPHIC_PARITY:
        plot_selection_error_curve(adjusted_predictor, ax=ax, show_plot=False)
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
