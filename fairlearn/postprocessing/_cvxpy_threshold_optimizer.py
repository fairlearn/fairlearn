# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Threshold optimizer with relaxed fairness constraints.

TODO
----
- Add option for constraining equality of positive predictions (independence
  criterion, aka demographic parity);
- Add option to use l1 or other distance functions for maximum tolerance between
  points (currently l-inf is in use).

"""
from __future__ import annotations

import logging
from itertools import product

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, MetaEstimatorMixin

# TODO: use this to validate input
# from fairlearn.utils._input_validation import _validate_and_reformat_input

from fairlearn.utils._common import _get_soft_predictions
from fairlearn.utils._common import unpack_fp_fn_costs

from ._cvxpy_utils import (
    compute_fair_optimum,
    ALL_CONSTRAINTS,
    NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE,
)
from ._roc_utils import (
    roc_convex_hull,
    calc_cost_of_point,
)

# TODO: try to use InterpolatedThreshold instead of our classifier API
from ._randomized_classifiers import (
    RandomizedClassifier,
    EnsembleGroupwiseClassifiers,
)


class _RelaxedThresholdOptimizer(BaseEstimator, MetaEstimatorMixin):
    r"""Class to encapsulate all the logic needed to compute the optimal
    postprocessing to fulfill fairness constraints with some optional
    tolerance.

    The method amounts to finding the set of (potentially randomized) 
    group-specific decision thresholds that maximize some objective (e.g., accuracy),
    given a maximum tolerance (or slack) on the fairness constraint fulfillment.

    This optimization problem amounts to a Linear Program (LP) as detailed in 
    :footcite:ct:`cruz2023reductions`. Solving the LP requires installing 
    `cvxpy`.

    Read more in the :ref:`User Guide <postprocessing>`.

    Parameters
    ----------
    predictor : object
        A prefit `scikit-learn compatible estimator <https://scikit-learn.org/stable/developers/develop.html#estimators>`_  # noqa
        whose output will be postprocessed.
        The predictor should output real-valued scores, as postprocessing 
        results will be extremely poor when performed over binarized 
        predictions.

    constraint : str, default='equalized_odds'
        Fairness constraint under which threshold optimization is performed. 
        Possible inputs currently are:

            'equalized_odds'
                match true positive and false positive rates across groups

    tolerance : float
        The absolute tolerance for the equalized odds fairness constraint.
        Will allow for at most `tolerance` distance between group-wise ROC 
        points (where distance is measured using l-infinity norm). Provided
        value must be in range [0, 1] (closed interval).


    objective_costs : dict, optional
        A dictionary detailing the cost for false positives and false negatives,
        of the form :code:`{'fp': <fp_cost>, 'fn': <fn_cost>}`. Will use the 0-1
        loss by default (maximum accuracy).

    grid_size : int, optional
        The maximum number of ticks (points) in each group's ROC curve, by
        default 1000. This corresponds to the maximum number of different 
        thresholds to use over a predictor.

    predict_method : {'auto', 'predict_proba', 'decision_function', 'predict'\
            }, default='auto'

        Defines which method of the ``estimator`` is used to get the output
        values.

            'auto'
                use one of :code:`predict_proba`, :code:`decision_function`, or 
                :code:`predict`, in that order.

            'predict_proba'
                use the second column from the output of :code:`predict_proba`. 
                It is assumed that the second column represents the positive 
                outcome.

            'decision_function'
                use the raw values given by the :code:`decision_function`.

            'predict'
                use the hard values reported by the :code:`predict` method if 
                estimator is a classifier, and the regression values if 
                estimator is a regressor.
                Warning: postprocessing may lead to poor results when using 
                :code:`predict_method='predict'` with classifiers, as that will
                binarize predictions.

    random_state : int, optional
        A random seed used for reproducibility when producing randomized
        classifiers, by default None (default: non-reproducible behavior).

    Raises
    ------
    ValueError
        A ValueError will be raised if constructor arguments are not valid.

    Notes
    -----
    The procedure for relaxed fairness constraint fulfillment is detailed in
    :footcite:ct:`cruz2023reductions`.

    The underlying threshold optimization algorithm is based on 
    :footcite:ct:`hardt2016equality`.

    This method is also implemented in its 
    `standalone Python package <https://github.com/socialfoundations/error-parity>`_.    # noqa

    """

    def __init__(
        self,
        *,
        predictor: BaseEstimator,
        constraint: str = "equalized_odds",
        tolerance: float = 0.0,
        objective_costs: dict = None,
        grid_size: int = 1000,
        predict_method: str = "auto",
        random_state: int = None,
    ):
        # Save arguments
        self.predictor = predictor
        self.constraint = constraint
        self.tolerance = tolerance
        self.objective_costs = objective_costs
        self.grid_size = grid_size
        self.predict_method = predict_method
        self.random_state = random_state

        # Validate constraint
        if self.constraint not in ALL_CONSTRAINTS:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        # Validate constraint tolerance
        if (
            not isinstance(self.tolerance, (float, int))
            or self.tolerance < 0
            or self.tolerance > 1
        ):
            raise ValueError(
                "Invalid `tolerance` provided: received "
                f"tolerance={self.tolerance}, but value should be in range "
                "[0, 1]."
            )

        # Check objective costs
        unpack_fp_fn_costs(self.objective_costs)

        # Initialize instance variables
        self._all_roc_data: dict = None
        self._all_roc_hulls: dict = None
        self._groupwise_roc_points: np.ndarray = None
        self._global_roc_point: np.ndarray = None
        self._global_prevalence: float = None
        self._realized_classifier: EnsembleGroupwiseClassifiers = None

    @property
    def false_pos_cost(self) -> np.ndarray:
        fp_cost, _fn_cost = unpack_fp_fn_costs(self.objective_costs)
        return fp_cost

    @property
    def false_neg_cost(self) -> np.ndarray:
        _fp_cost, fn_cost = unpack_fp_fn_costs(self.objective_costs)
        return fn_cost

    @property
    def groupwise_roc_points(self) -> np.ndarray:
        return self._groupwise_roc_points

    @property
    def global_roc_point(self) -> np.ndarray:
        return self._global_roc_point

    def cost(
        self,
        *,
        false_pos_cost: float = 1.0,
        false_neg_cost: float = 1.0,
    ) -> float:
        """Compute the theoretical cost of the solution found.

        Use false_pos_cost=false_neg_cost=1 for the 0-1 loss (the standard error
        rate), which amounts to maximizing accuracy.

        You can find the cost realized from the LP optimization by calling:
        >>> obj.cost(
        >>>     false_pos_cost=obj.false_pos_cost,
        >>>     false_neg_cost=obj.false_neg_cost,
        >>> )

        Parameters
        ----------
        false_pos_cost : float, optional
            The cost of a FALSE POSITIVE error, by default 1.
        false_neg_cost : float, optional
            The cost of a FALSE NEGATIVE error, by default 1.

        Returns
        -------
        float
            The cost of the solution found.
        """
        self._check_fit_status()
        global_fpr, global_tpr = self.global_roc_point

        return calc_cost_of_point(
            fpr=global_fpr,
            fnr=1 - global_tpr,
            prevalence=self._global_prevalence,
            false_pos_cost=false_pos_cost,
            false_neg_cost=false_neg_cost,
        )

    def constraint_violation(self) -> float:
        """Constraint violation of the LP solution found.

        Returns
        -------
        float
            The fairness constraint violation.
        """
        self._check_fit_status()

        if self.constraint not in ALL_CONSTRAINTS:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        if self.constraint == "equalized_odds":
            return self.equalized_odds_violation()

        elif self.constraint.endswith("rate_parity"):
            constraint_to_error_type = {
                "true_positive_rate_parity": "fn",
                "false_positive_rate_parity": "fp",
                "true_negative_rate_parity": "fp",
                "false_negative_rate_parity": "fn",
            }

            return self.error_rate_parity_constraint_violation(
                error_type=constraint_to_error_type[self.constraint],
            )

        else:
            raise NotImplementedError(
                "Standalone constraint violation not yet computed for "
                f"constraint='{self.constraint}'."
            )

    def error_rate_parity_constraint_violation(self, error_type: str) -> float:
        """Compute the theoretical violation of an error-rate parity constraint.

        Parameters
        ----------
        error_type : str
            One of the following values:
                "fp", for false positive errors (FPR or TNR parity);
                "fn", for false negative errors (TPR or FNR parity).

        Returns
        -------
        float
            The maximum constraint violation among all groups.
        """
        self._check_fit_status()
        valid_error_types = ("fp", "fn")
        if error_type not in valid_error_types:
            raise ValueError(
                f"Invalid error_type='{error_type}', must be one of "
                f"{valid_error_types}."
            )

        roc_idx_of_interest = 0 if error_type == "fp" else 1

        return self._max_l_inf_between_points(
            points=[
                roc_point[roc_idx_of_interest]
                for roc_point in self.groupwise_roc_points
            ],
        )

    def equalized_odds_violation(self) -> float:
        """Compute the theoretical violation of the equalized odds constraint.

        That is, the maximum l-inf distance between the ROC point of any pair
        of groups.

        Returns
        -------
        float
            The equal-odds constraint violation.
        """
        self._check_fit_status()

        # Compute l-inf distance between each pair of groups
        return self._max_l_inf_between_points(
            points=self.groupwise_roc_points,
        )

    @staticmethod
    def _max_l_inf_between_points(points: list[float | np.ndarray]) -> float:
        # Number of points (should correspond to the number of groups)
        n_points = len(points)

        # Compute l-inf distance between each pair of groups
        l_inf_constraint_violation = [
            (np.linalg.norm(points[i] - points[j], ord=np.inf), (i, j))
            for i, j in product(range(n_points), range(n_points))
            if i < j
        ]

        # Return the maximum
        max_violation, (groupA, groupB) = max(l_inf_constraint_violation)
        logging.info(
            (
                "Maximum fairness violation is between "
                "group=%d (p=%s) and "
                "group=%d (p=%s);"
            ),
            groupA,
            points[groupA],
            groupB,
            points[groupB],
        )

        return max_violation

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sensitive_features: np.ndarray,  # TODO validate input and convert to proper format
        y_scores: np.ndarray = None,
    ):
        """Find the optimal fair postprocessing.

        That is, it finds the postprocessing that minimizes loss, while
        fulfilling the (possibly relaxed) fairness constraint on the provided
        data.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The input labels.
        sensitive_features : np.ndarray
            The sensitive features (group membership) of each sample.
            Assumes groups are numbered [0, 1, ..., num_groups-1].
        y_scores : np.ndarray, optional
            The pre-computed model predictions on this data.

        Returns
        -------
        callable
            Returns self.
        """
        # Compute group stats
        self._global_prevalence = np.sum(y) / len(y)

        unique_groups = np.unique(sensitive_features)
        num_groups = len(unique_groups)
        if np.max(unique_groups) > num_groups - 1:
            raise ValueError(
                "Groups should be numbered starting at 0, and up to "
                f"num_groups-1. Got {num_groups} groups, but max value is "
                f"{np.max(unique_groups)} != num_groups-1 == {num_groups-1}."
            )

        # Relative group sizes for LN and LP samples
        group_sizes_label_neg = np.array(
            [np.sum(1 - y[sensitive_features == g]) for g in unique_groups]
        )
        group_sizes_label_pos = np.array(
            [np.sum(y[sensitive_features == g]) for g in unique_groups]
        )

        if np.sum(group_sizes_label_neg) + np.sum(group_sizes_label_pos) != len(y):
            raise RuntimeError(
                "Failed input validation. Are you using non-binary labels?"
            )

        # Convert to relative sizes
        group_sizes_label_neg = group_sizes_label_neg.astype(float) / np.sum(
            group_sizes_label_neg
        )
        group_sizes_label_pos = group_sizes_label_pos.astype(float) / np.sum(
            group_sizes_label_pos
        )

        # Compute group-wise ROC curves
        if y_scores is None:
            y_scores = _get_soft_predictions(self.predictor, X, self.predict_method)

        self._all_roc_data = dict()
        for g in unique_groups:
            group_filter = sensitive_features == g

            roc_curve_data = roc_curve(
                y[group_filter],
                y_scores[group_filter],
            )

            # Check if max_roc_ticks is exceeded
            fpr, tpr, thrs = roc_curve_data
            if self.grid_size is not None and len(fpr) > self.grid_size:
                indices_to_keep = np.arange(
                    0, len(fpr), len(fpr) / self.grid_size
                ).astype(int)

                # Bottom-left (0,0) and top-right (1,1) points must be kept
                indices_to_keep[-1] = len(fpr) - 1
                roc_curve_data = (
                    fpr[indices_to_keep],
                    tpr[indices_to_keep],
                    thrs[indices_to_keep],
                )

            self._all_roc_data[g] = roc_curve_data

        # Compute convex hull of each ROC curve
        self._all_roc_hulls = dict()
        for g in unique_groups:
            group_fpr, group_tpr, _group_thresholds = self._all_roc_data[g]

            curr_roc_points = np.stack((group_fpr, group_tpr), axis=1)
            curr_roc_points = np.vstack(
                (curr_roc_points, [1, 0])
            )  # Add point (1, 0) to ROC curve

            self._all_roc_hulls[g] = roc_convex_hull(curr_roc_points)

        # Find the group-wise optima that fulfill the fairness criteria
        self._groupwise_roc_points, self._global_roc_point = compute_fair_optimum(
            fairness_constraint=self.constraint,
            groupwise_roc_hulls=self._all_roc_hulls,
            tolerance=self.tolerance,
            group_sizes_label_pos=group_sizes_label_pos,
            group_sizes_label_neg=group_sizes_label_neg,
            global_prevalence=self._global_prevalence,
            false_positive_cost=self.false_pos_cost,
            false_negative_cost=self.false_neg_cost,
        )

        # Construct each group-specific classifier
        all_rand_clfs = {
            g: RandomizedClassifier.construct_at_target_ROC(  # TODO: check InterpolatedThresholder
                predictor=self.predictor,
                roc_curve_data=self._all_roc_data[g],
                target_roc_point=self._groupwise_roc_points[g],
                seed=self.random_state,
            )
            for g in unique_groups
        }

        # Construct the global classifier (can be used for all groups)
        # TODO: check InterpolatedThresholder
        self._realized_classifier = EnsembleGroupwiseClassifiers(
            group_to_clf=all_rand_clfs
        )
        return self

    def _check_fit_status(self, raise_error: bool = True) -> bool:
        """Check whether this classifier has been fit on some data.

        Parameters
        ----------
        raise_error : bool, optional
            Whether to raise an error if the classifier is uninitialized
            (otherwise will just return False), by default True.

        Returns
        -------
        is_fit : bool
            Whether the classifier was already fit on some data.

        Raises
        ------
        RuntimeError
            If `raise_error==True`, raises an error if the classifier is
            uninitialized.
        """
        if self._realized_classifier is None:
            if not raise_error:
                return False

            raise RuntimeError(
                "This classifier has not yet been fitted to any data. "
                "Call clf.fit(...) before this method."
            )

        return True

    def predict(self, X: np.ndarray, *, sensitive_features: np.ndarray) -> np.ndarray:
        return self._realized_classifier(X, sensitive_features=sensitive_features)
