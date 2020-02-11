# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import pytest
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing.plotting import plot_selection_error_curve, plot_roc_curve
from fairlearn.postprocessing._constants import DEMOGRAPHIC_PARITY, EQUALIZED_ODDS

from .conftest import scores_ex, ExamplePredictor, is_invalid_transformation


@pytest.mark.mpl_image_compare()
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_plot_equalized_odds(data_X_y_sf):
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(scores_ex),
                                            constraints=EQUALIZED_ODDS)
    adjusted_predictor.fit(data_X_y_sf.X, data_X_y_sf.y,
                           sensitive_features=data_X_y_sf.sensitive_features)
    fig = plt.figure()
    plot_roc_curve(adjusted_predictor, show_plot=False)
    return fig


@pytest.mark.mpl_image_compare()
@pytest.mark.uncollect_if(func=is_invalid_transformation)
def test_plot_demographic_parity(data_X_y_sf):
    adjusted_predictor = ThresholdOptimizer(unconstrained_predictor=ExamplePredictor(scores_ex),
                                            constraints=DEMOGRAPHIC_PARITY)
    adjusted_predictor.fit(data_X_y_sf.X, data_X_y_sf.y,
                           sensitive_features=data_X_y_sf.sensitive_features)
    fig = plt.figure()
    plot_selection_error_curve(adjusted_predictor, show_plot=False)
    return fig
