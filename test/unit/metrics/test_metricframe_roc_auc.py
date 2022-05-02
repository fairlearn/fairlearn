# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
import numpy as np
import pkg_resources
import pytest
from fairlearn.metrics import RocAuc
from .data_for_test import y_t, g_1, g_2, y_score

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


def _plot_by_group(y_true, y_score, sensitive_features):
    import matplotlib.pyplot as plt
    rc = RocAuc(
        y_true=y_true,
        y_score=y_score,
        sensitive_features=sensitive_features)
    roc_plot = rc.plot_by_group()
    return roc_plot.figure_


def is_mpl_installed():
    try:
        pkg_resources.get_distribution("pytest-mpl")
        pkg_resources.get_distribution("matplotlib")
        return True
    except pkg_resources.DistributionNotFound:
        return False


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
class TestPlots:
    @pytest.mark.mpl_image_compare(filename="plot_by_group_ex1.png")
    def test_plot_by_group_ex1(self):
        sensitive_features = np.hstack((g_1.reshape(-1,1), g_2.reshape(-1,1)))
        return _plot_by_group(y_t, y_score, sensitive_features)

    # @pytest.mark.mpl_image_compare(filename="equalized_odds_ex2.png")
    # def test_plot_equalized_odds_ex2(self):
    #     return _fit_and_plot('equalized_odds', _data_ex2)

    # @pytest.mark.mpl_image_compare(filename="equalized_odds_ex3.png")
    # def test_plot_equalized_odds_ex3(self):
    #     return _fit_and_plot('equalized_odds', _data_ex3)

    # @pytest.mark.mpl_image_compare(filename="demographic_parity_ex1.png")
    # def test_plot_demographic_parity_ex1(self):
    #     return _fit_and_plot('demographic_parity', _data_ex1)

    # @pytest.mark.mpl_image_compare(filename="demographic_parity_ex2.png")
    # def test_plot_demographic_parity_ex2(self):
    #     return _fit_and_plot('demographic_parity', _data_ex2)

    # @pytest.mark.mpl_image_compare(filename="demographic_parity_ex3.png")
    # def test_plot_demographic_parity_ex3(self):
    #     return _fit_and_plot('demographic_parity', _data_ex3)
