# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from __future__ import annotations

import pandas as pd
from numpy import asarray, zeros
from sklearn.metrics import RocCurveDisplay

from ..postprocessing._constants import _MATPLOTLIB_IMPORT_ERROR_MESSAGE
from ..utils._input_validation import (
    _INCONSISTENT_ARRAY_LENGTH,
    _validate_and_reformat_input,
)

_CHANCE_LEVEL_LABEL = "Chance level (AUC = 0.50)"
_OVERALL_LABEL = "Overall"


def plot_roc_curve_by_group(
    y_true,
    y_score,
    *,
    sensitive_features,
    ax=None,
    plot_overall=True,
    plot_chance_level=True,
    pos_label=None,
    title=None,
):
    r"""Plot ROC curves for a binary classifier disaggregated by sensitive feature.

    A separate Receiver Operating Characteristic (ROC) curve is drawn for each
    subgroup defined by ``sensitive_features``, together with the area under the
    curve (AUC) for that subgroup. Comparing the curves makes it easy to see
    whether the classifier separates the positive and negative classes equally
    well across groups: curves that lie on top of one another indicate similar
    ranking performance, whereas curves that diverge indicate that the model
    discriminates between the classes better for some groups than for others.

    When more than one sensitive feature is provided, the unique combinations of
    their values define the subgroups (for example ``"Female,White"``), following
    the same convention as :class:`~fairlearn.metrics.MetricFrame`.

    .. versionadded:: 0.15.0

    Parameters
    ----------
    y_true : array-like
        The ground-truth binary labels.

    y_score : array-like
        Target scores for the positive class, such as probability estimates
        returned by ``predict_proba(X)[:, 1]`` or the output of
        ``decision_function``.

    sensitive_features : array-like, dict of 1d arrays, pandas.DataFrame
        The sensitive feature(s) used to define the subgroups. At least one
        sensitive feature must be provided.

    ax : matplotlib.axes.Axes, optional
        The axes on which to draw. If not supplied, a new figure and axes are
        created.

    plot_overall : bool, default=True
        Whether to also plot the ROC curve computed over the entire dataset for
        comparison.

    plot_chance_level : bool, default=True
        Whether to plot the chance-level (no-skill) diagonal that a random
        classifier would produce.

    pos_label : int, float, bool or str, optional
        The label of the positive class. When ``None``, the positive class is
        inferred by scikit-learn (1 when the labels are in ``{-1, 1}`` or
        ``{0, 1}``).

    title : str, optional
        The title to set on the axes, for example the name of the estimator.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the curves were drawn.

    Notes
    -----
    This function only produces the plot. To obtain the AUC scores
    programmatically, use :class:`~fairlearn.metrics.MetricFrame` directly, which
    is the idiomatic way to disaggregate any metric in Fairlearn::

        from sklearn.metrics import roc_auc_score
        from fairlearn.metrics import MetricFrame

        mf = MetricFrame(
            metrics=roc_auc_score,
            y_true=y_true,
            y_pred=y_score,
            sensitive_features=sensitive_features,
        )
        mf.by_group  # AUC per subgroup
        mf.overall   # AUC over the whole dataset

    Each subgroup must contain both classes for its ROC curve and AUC to be
    well defined. To further customize the appearance of the plot, pass an
    existing :class:`matplotlib.axes.Axes` via ``ax`` or restyle the returned
    Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    y_true = asarray(y_true)
    y_score = asarray(y_score)
    if y_score.shape[0] != y_true.shape[0]:
        raise ValueError(_INCONSISTENT_ARRAY_LENGTH.format("y_true and y_score"))

    # Validate and merge the sensitive feature(s) into a single Series of group
    # names without coercing y_true, so that non-numeric class labels remain
    # usable together with pos_label.
    if isinstance(sensitive_features, dict):
        sensitive_features = pd.DataFrame(sensitive_features)
    *_, sensitive_features, _ = _validate_and_reformat_input(
        zeros((len(y_true), 1)),  # dummy X; only sensitive_features is needed
        expect_y=False,
        sensitive_features=sensitive_features,
    )

    if ax is None:
        _, ax = plt.subplots()

    if plot_chance_level:
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.8", label=_CHANCE_LEVEL_LABEL)

    if plot_overall:
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            name=_OVERALL_LABEL,
            pos_label=pos_label,
            ax=ax,
        )

    for group in sorted(sensitive_features.unique()):
        mask = (sensitive_features == group).to_numpy()
        RocCurveDisplay.from_predictions(
            y_true[mask],
            y_score[mask],
            name=str(group),
            pos_label=pos_label,
            ax=ax,
        )

    if title is not None:
        ax.set_title(title)

    return ax
