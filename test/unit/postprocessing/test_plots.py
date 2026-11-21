# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import pytest

from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer

from .conftest import ExamplePredictor, _data_ex1, _data_ex2, _data_ex3, scores_ex

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


@pytest.fixture()
def close_figs():
    # Close before and after each test to avoid lingering figures between tests
    import matplotlib.pyplot as plt

    plt.close("all")  # pre-test (important for backend switches)
    yield
    plt.close("all")  # post-test cleanup


def _fit_and_plot(constraints, plotting_data, tol: float | None = None):
    import matplotlib.pyplot as plt

    adjusted_predictor = ThresholdOptimizer(
        estimator=ExamplePredictor(scores_ex),
        constraints=constraints,
        predict_method="predict",
        tol=tol,
    )
    adjusted_predictor.fit(
        plotting_data.X,
        plotting_data.y,
        sensitive_features=plotting_data.sensitive_features,
    )
    fig, (ax) = plt.subplots(1, 1)
    plot_threshold_optimizer(adjusted_predictor, ax=ax, show_plot=False)
    return fig


def is_mpl_installed():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import pytest_mpl  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def _baseline_dir():
    # We keep two committed baseline sets because matplotlib 3.11 dropped Python
    # 3.10 wheels and rendered output differently from the 3.10.x line, which
    # produces large RMS differences (~20-37 vs. tolerance 2) against a single
    # set of baselines. Pick the matching directory at import time.
    if not is_mpl_installed():
        return "../plot_snapshots"
    import matplotlib
    from packaging.version import Version

    if Version(matplotlib.__version__) >= Version("3.11"):
        return "../plot_snapshots"
    return "../plot_snapshots_mpl310"


_BASELINE_DIR = _baseline_dir()


@pytest.mark.usefixtures("close_figs")
@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
class TestPlots:
    @pytest.mark.mpl_image_compare(
        filename="post_processing_equalized_odds_ex1.png", baseline_dir=_BASELINE_DIR
    )
    def test_plot_equalized_odds_ex1(self):
        return _fit_and_plot("equalized_odds", _data_ex1)

    @pytest.mark.mpl_image_compare(
        filename="post_processing_equalized_odds_ex2.png", baseline_dir=_BASELINE_DIR
    )
    def test_plot_equalized_odds_ex2(self):
        return _fit_and_plot("equalized_odds", _data_ex2)

    @pytest.mark.mpl_image_compare(
        filename="post_processing_equalized_odds_ex3.png", baseline_dir=_BASELINE_DIR
    )
    def test_plot_equalized_odds_ex3(self):
        return _fit_and_plot("equalized_odds", _data_ex3)

    @pytest.mark.mpl_image_compare(
        filename="post_processing_demographic_parity_ex1.png", baseline_dir=_BASELINE_DIR
    )
    def test_plot_demographic_parity_ex1(self):
        return _fit_and_plot("demographic_parity", _data_ex1)

    @pytest.mark.mpl_image_compare(
        filename="post_processing_demographic_parity_ex2.png", baseline_dir=_BASELINE_DIR
    )
    def test_plot_demographic_parity_ex2(self):
        return _fit_and_plot("demographic_parity", _data_ex2)

    @pytest.mark.mpl_image_compare(
        filename="post_processing_demographic_parity_ex3.png", baseline_dir=_BASELINE_DIR
    )
    def test_plot_demographic_parity_ex3(self):
        return _fit_and_plot("demographic_parity", _data_ex3)

    @pytest.mark.mpl_image_compare(
        filename="post_processing_demographic_parity_tol.png", baseline_dir=_BASELINE_DIR
    )
    def test_plot_demographic_parity_tol(self):
        return _fit_and_plot("demographic_parity", _data_ex1, tol=0.1)

    @pytest.mark.parametrize("constrained_layout", [False, True])
    @pytest.mark.parametrize(
        "plotting_data", [_data_ex1, _data_ex3], ids=["three-groups", "five-groups"]
    )
    def test_plot_demographic_parity_tol_legend_is_outside_axes(
        self, plotting_data, constrained_layout
    ):
        import matplotlib.pyplot as plt

        with plt.rc_context({"figure.constrained_layout.use": constrained_layout}):
            fig = _fit_and_plot("demographic_parity", plotting_data, tol=0.1)
            fig.canvas.draw()

        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        legend_bounds = ax.get_legend().get_window_extent(renderer)
        axes_bounds = ax.get_window_extent(renderer)

        assert legend_bounds.y1 < axes_bounds.y0
        assert fig.bbox.contains(legend_bounds.x0, legend_bounds.y0)
        assert fig.bbox.contains(legend_bounds.x1, legend_bounds.y1)
        assert fig.get_constrained_layout() == constrained_layout
