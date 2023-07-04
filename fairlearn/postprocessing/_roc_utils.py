# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Helper functions for threshold optimization methods.

NOTE
----
- Most utils defined here likely have a similar counter-part already implemented
  somewhere in the `fairlearn` code-base.
- With time they will probably be substituted by that counter-part, and these
  implementations removed.
"""
import logging
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix


def calc_cost_of_point(
    fpr: float,
    fnr: float,
    prevalence: float,
    *,
    false_pos_cost: float = 1.0,
    false_neg_cost: float = 1.0,
) -> float:
    """Calculate the cost of the given ROC point.

    Parameters
    ----------
    fpr : float
        The false positive rate (FPR).
    fnr : float
        The false negative rate (FNR).
    prevalence : float
        The prevalence of positive samples in the dataset,
        i.e., np.sum(y_true) / len(y_true)
    false_pos_cost : float, optional
        The cost of a false positive error, by default 1.
    false_neg_cost : float, optional
        The cost of a false negative error, by default 1.

    Returns
    -------
    cost : float
        The cost of the given ROC point (divided by the size of the dataset).
    """
    cost_vector = np.array([false_pos_cost, false_neg_cost])
    weight_vector = np.array([1 - prevalence, prevalence])
    return cost_vector * weight_vector @ np.array([fpr, fnr])


def compute_roc_point_from_predictions(y_true, y_pred_binary):
    """Compute the ROC point associated with the provided binary predictions.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred_binary : np.ndarray
        The binary predictions.

    Returns
    -------
    tuple[float, float]
        The resulting ROC point, i.e., a tuple (FPR, TPR).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    # FPR = FP / LN
    fpr = fp / (fp + tn)

    # TPR = TP / LP
    tpr = tp / (tp + fn)

    return (fpr, tpr)


def compute_global_roc_from_groupwise(
    groupwise_roc_points: np.ndarray,
    groupwise_label_pos_weight: np.ndarray,
    groupwise_label_neg_weight: np.ndarray,
) -> np.ndarray:
    """Compute the global ROC point that corresponds to the provided group-wise ROC points.

    The global ROC is a linear combination of the group-wise points, with
    different weights for computing FPR and TPR -- the first related to LNs, and
    the second to LPs.

    Parameters
    ----------
    groupwise_roc_points : np.ndarray
        An array of shape (n_groups, n_roc_dims) containing one ROC point per
        group.
    groupwise_label_pos_weight : np.ndarray
        The relative size of each group in terms of its label POSITIVE samples
        (out of all POSITIVE samples, how many are in each group).
    groupwise_label_neg_weight : np.ndarray
        The relative size of each group in terms of its label NEGATIVE samples
        (out of all NEGATIVE samples, how many are in each group).

    Returns
    -------
    global_roc_point : np.ndarray
        A single point that corresponds to the global outcome of the given
        group-wise ROC points.
    """
    n_groups, _ = groupwise_roc_points.shape

    # Validating input shapes
    if (
        len(groupwise_label_pos_weight) != len(groupwise_label_neg_weight)
        or len(groupwise_label_pos_weight) != n_groups
    ):
        raise ValueError(
            "Invalid input shapes: length of all arguments must be equal (the "
            "number of different sensitive groups)."
        )

    # Normalize group LP (/LN) weights by their size
    if not np.isclose(groupwise_label_pos_weight.sum(), 1.0):
        groupwise_label_pos_weight /= groupwise_label_pos_weight.sum()
    if not np.isclose(groupwise_label_neg_weight.sum(), 1.0):
        groupwise_label_neg_weight /= groupwise_label_neg_weight.sum()

    # Compute global FPR (weighted by relative number of LNs in each group)
    global_fpr = groupwise_label_neg_weight @ groupwise_roc_points[:, 0]

    # Compute global TPR (weighted by relative number of LPs in each group)
    global_tpr = groupwise_label_pos_weight @ groupwise_roc_points[:, 1]

    global_roc_point = np.array([global_fpr, global_tpr])
    return global_roc_point


def roc_convex_hull(roc_points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of the provided ROC points.

    Parameters
    ----------
    roc_points : np.ndarray
        An array of shape (n_points, n_dims) containing all points
        of a provided ROC curve.

    Returns
    -------
    hull_points : np.ndarray
        An array of shape (n_hull_points, n_dim) containing all
        points in the convex hull of the ROC curve.
    """
    # Save init data just for logging
    init_num_points, _dims = roc_points.shape

    # Compute convex hull
    hull = ConvexHull(roc_points)

    # NOTE: discarding points below the diagonal seems to lead to bugs later on, idk why...
    # Discard points in the interior of the convex hull,
    # and other useless points (below main diagonal)
    # points_above_diagonal = np.argwhere(roc_points[:, 1] >= roc_points[:, 0]).ravel()
    # hull_indices = sorted(set(hull.vertices) & set(points_above_diagonal))

    hull_indices = hull.vertices

    logging.info(
        "ROC convex hull contains %.1f%% of the original points.",
        (len(hull_indices) / init_num_points) * 100,
    )

    return roc_points[hull_indices]
