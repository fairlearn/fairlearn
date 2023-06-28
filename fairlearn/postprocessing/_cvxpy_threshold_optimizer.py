"""Threshold optimizer with relaxed fairness constraints.

TODO
----
- Add option for constraining only equality of FPR or TPR (currently it must be 
both -> equal odds);
- Add option for constraining equality of positive predictions (independence
criterion, aka demographic parity);
- Add option to use l1 or linf distances for maximum tolerance between points.
  - Currently 'equal_odds' is defined using l-infinity distance (max between
  TPR and FPR distances);

"""
import logging
from itertools import product
from typing import Callable

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from fairlearn.utils._input_validation import _validate_and_reformat_input
from fairlearn.reductions._moments.error_rate import _MESSAGE_BAD_COSTS

from ._cvxpy_utils import compute_equal_odds_optimum
from ._roc_utils import (
    roc_convex_hull,
    calc_cost_of_point,
)
from ._randomized_classifiers import (  # TODO: try to use fairlearn's InterpolatedThreshold instead of our classifier API
    RandomizedClassifier,
    EnsembleGroupwiseClassifiers,
)


class _RelaxedThresholdOptimizer(BaseEstimator, MetaEstimatorMixin):
    r"""Class to encapsulate all the logic needed to compute the optimal
    postprocessing to fulfill fairness constraints with some optional
    tolerance.

    The method amounts to finding the set of (potentially randomized) 
    group-specific decision thresholds that maximize some goal (e.g., accuracy),
    given a maximum tolerance (or slack) on the fairness constraint fulfillment.
    
    This optimization problem amounts to a Linear Program (LP) as detailed in 
    [1]_. Solving the LP requires installing `cvxpy`.

    Read more in the :ref:`User Guide <postprocessing>`.

    Parameters
    ----------
    estimator : object
        A `scikit-learn compatible estimator <https://scikit-learn.org/stable/developers/develop.html#estimators>`_  # noqa
        whose output will be postprocessed.
        The estimator should output real-valued scores, as postprocessing 
        results will be extremely poor when performed over binarized 
        predictions.

    tolerance : float
        The absolute tolerance for the equal odds fairness constraint.
        Will allow for `tolerance` difference between group-wise ROC points.

    objective_costs : dict
        A dictionary detailing the cost for false positives and false negatives,
        of the form :code:`{'fp': <fp_cost>, 'fn': <fn_cost>}`. Will use the 0-1
        loss by default (maximum accuracy).

    grid_size : int, optional
        The maximum number of ticks (points) in each group's ROC curve, by
        default 1000. This corresponds to the maximum number of different 
        thresholds to use over a predictor.

    seed : int
        A random seed used for reproducibility when producing randomized
        classifiers, by default None (default: non-reproducible behavior).

    Notes
    -----
    The procedure for relaxed fairness constraint fulfillment is detailed in
    `Cruz et al. (2023) <https://arxiv.org/abs/2306.07261>`_ [1]_.
    The underlying threshold optimization algorithm is based on 
    `Hardt et al. (2016) <https://arxiv.org/abs/1610.02413>`_ [2]_.

    References
    ----------
    .. [1] A. Cruz, and M. Hardt, "Unprocessing Seven Years of 
       Algorithmic Fairness," arXiv.org, 15-Jun-2023.
       [Online]. Available: https://arxiv.org/abs/2306.07261.

    .. [2] M. Hardt, E. Price, and N. Srebro, "Equality of Opportunity in
       Supervised Learning," arXiv.org, 07-Oct-2016.
       [Online]. Available: https://arxiv.org/abs/1610.02413.

    """

    def __init__(
            self,
            estimator: BaseEstimator,
            tolerance: float,
            objective_costs: dict = None,
            grid_size: int = 1000,
            seed: int = None,
        ):

        # Save arguments
        self.estimator = estimator
        self.tolerance = tolerance
        self.max_grid_size = grid_size

        # Unpack objective costs
        if objective_costs is None:
            self.false_pos_cost = 1.0
            self.false_neg_cost = 1.0
        else:
            self.false_pos_cost, self.false_neg_cost = \
                self.unpack_objective_costs(objective_costs)

        # Randomly sample a seed if none was provided
        self.seed = np.random.randint(2 ** 20)

        # Initialize instance variables
        self._all_roc_data: dict = None
        self._all_roc_hulls: dict = None
        self._groupwise_roc_points: np.ndarray = None
        self._global_roc_point: np.ndarray = None
        self._global_prevalence: float = None
        self._realized_classifier: EnsembleGroupwiseClassifiers = None

    @staticmethod
    def unpack_objective_costs(objective_costs: dict) -> tuple[float, float]:
        """Validates and unpacks the given `objective_costs`.

        Parameters
        ----------
        objective_costs : dict
            A dictionary detailing the cost for false positives and false negatives,
            of the form :code:`{'fp': <fp_cost>, 'fn': <fn_cost>}`. Will use the 0-1
            loss by default (maximum accuracy).
            
        Returns
        -------
        tuple[float, float]
            A tuple respectively composed of the cost of false positives and the
            cost of false negatives, i.e., a tuple with 
            :code:`(fp_cost, fn_cost)`.

        Raises
        ------
        ValueError
            Raised when the provided costs are invalid (e.g., missing keys
            in the provided dict, or negative costs).
        """
        if (
            type(objective_costs) is dict
            and objective_costs.keys() == {"fp", "fn"}
            and objective_costs["fp"] >= 0.0
            and objective_costs["fn"] >= 0.0
            and objective_costs["fp"] + objective_costs["fn"] > 0.0
        ):
            fp_cost = objective_costs["fp"]
            fn_cost = objective_costs["fn"]
        else:
            raise ValueError(_MESSAGE_BAD_COSTS)
        
        return fp_cost, fn_cost

    @property
    def groupwise_roc_points(self) -> np.ndarray:
        return self._groupwise_roc_points

    @property
    def global_roc_point(self) -> np.ndarray:
        return self._global_roc_point

    def cost(
            self,
            false_pos_cost: float = None,
            false_neg_cost: float = None,
        ) -> float:
        """Computes the theoretical cost of the solution found.

        Use false_pos_cost==false_neg_cost==1 for the 0-1 loss (the 
        standard error rate), which amounts to maximizing accuracy.

        Parameters
        ----------
        false_pos_cost : float, optional
            The cost of a FALSE POSITIVE error, by default will take the value
            given in the object's constructor.
        false_neg_cost : float, optional
            The cost of a FALSE NEGATIVE error, by default will take the value
            given in the object's constructor.

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
            false_pos_cost=false_pos_cost or self.false_pos_cost,
            false_neg_cost=false_neg_cost or self.false_neg_cost,
        )
    
    def constraint_violation(self) -> float:
        """This method should be part of a common interface between different
        relaxed-constraint classes.

        Returns
        -------
        float
            The fairness constraint violation.
        """
        return self.equal_odds_violation()

    def equal_odds_violation(self) -> float:
        """Computes the theoretical violation of the equal odds constraint 
        (i.e., the maximum l-inf distance between the ROC point of any pair
        of groups).

        Returns
        -------
        float
            The equal-odds constraint violation.
        """
        self._check_fit_status()

        n_groups = len(self.groupwise_roc_points)

        # Compute l-inf distance between each pair of groups
        linf_constraint_violation = [
            (np.linalg.norm(
                self.groupwise_roc_points[i] - self.groupwise_roc_points[j],
                ord=np.inf), (i, j))
            for i, j in product(range(n_groups), range(n_groups))
            if i < j
        ]

        # Return the maximum
        max_violation, (groupA, groupB) = max(linf_constraint_violation)
        logging.info(
            f"Maximum fairness violation is between "
            f"group={groupA} (p={self.groupwise_roc_points[groupA]}) and "
            f"group={groupB} (p={self.groupwise_roc_points[groupB]});"
        )

        return max_violation


    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray, y_scores: np.ndarray = None):
        """Fit this predictor to achieve the (possibly relaxed) equal odds 
        constraint on the provided data.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The input labels.
        group : np.ndarray
            The group membership of each sample.
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

        unique_groups = np.unique(group)
        num_groups = len(unique_groups)
        if np.max(unique_groups) > num_groups-1:
            raise ValueError(
                f"Groups should be numbered starting at 0, and up to "
                f"num_groups-1. Got {num_groups} groups, but max value is "
                f"{np.max(unique_groups)} != num_groups-1 == {num_groups-1}."
            )

        # Relative group sizes for LN and LP samples
        group_sizes_label_neg = np.array([
            np.sum(1 - y[group == g]) for g in unique_groups
        ])
        group_sizes_label_pos = np.array([
            np.sum(y[group == g]) for g in unique_groups
        ])

        if np.sum(group_sizes_label_neg) + np.sum(group_sizes_label_pos) != len(y):
            raise RuntimeError(f"Failed sanity check. Are you using non-binary labels?")

        # Convert to relative sizes
        group_sizes_label_neg = group_sizes_label_neg.astype(float) / np.sum(group_sizes_label_neg)
        group_sizes_label_pos = group_sizes_label_pos.astype(float) / np.sum(group_sizes_label_pos)

        # Compute group-wise ROC curves
        if y_scores is None:
            y_scores = self.estimator(X)

        self._all_roc_data = dict()
        for g in unique_groups:
            group_filter = group == g

            roc_curve_data = roc_curve(
                y[group_filter],
                y_scores[group_filter],
            )

            # Check if max_roc_ticks is exceeded
            fpr, tpr, thrs = roc_curve_data
            if self.max_grid_size is not None and len(fpr) > self.max_grid_size:
                indices_to_keep = np.arange(0, len(fpr), len(fpr) / self.max_grid_size).astype(int)

                # Bottom-left (0,0) and top-right (1,1) points must be kept
                indices_to_keep[-1] = len(fpr) - 1
                roc_curve_data = (fpr[indices_to_keep], tpr[indices_to_keep], thrs[indices_to_keep])

            self._all_roc_data[g] = roc_curve_data

        # Compute convex hull of each ROC curve
        self._all_roc_hulls = dict()
        for g in unique_groups:
            group_fpr, group_tpr, _group_thresholds = self._all_roc_data[g]

            curr_roc_points = np.stack((group_fpr, group_tpr), axis=1)
            curr_roc_points = np.vstack((curr_roc_points, [1, 0]))  # Add point (1, 0) to ROC curve

            self._all_roc_hulls[g] = roc_convex_hull(curr_roc_points)

        # Find the group-wise optima that fulfill the fairness criteria
        self._groupwise_roc_points, self._global_roc_point = compute_equal_odds_optimum(
            groupwise_roc_hulls=self._all_roc_hulls,
            fairness_tolerance=self.tolerance,
            group_sizes_label_pos=group_sizes_label_pos,
            group_sizes_label_neg=group_sizes_label_neg,
            global_prevalence=self._global_prevalence,
            false_positive_cost=self.false_pos_cost,
            false_negative_cost=self.false_neg_cost,
        )

        # Construct each group-specific classifier
        all_rand_clfs = {
            g: RandomizedClassifier.construct_at_target_ROC(
                predictor=self.estimator,
                roc_curve_data=self._all_roc_data[g],
                target_roc_point=self._groupwise_roc_points[g],
                seed=self.seed,
            )
            for g in unique_groups
        }

        # Construct the global classifier (can be used for all groups)
        self._realized_classifier = EnsembleGroupwiseClassifiers(group_to_clf=all_rand_clfs)
        return self
    
    def _check_fit_status(self, raise_error: bool = True) -> bool:
        """Checks whether this classifier has been fit on some data.
        
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
                "Call clf.fit(...) before this method.")

        return True

    def predict(self, X: np.ndarray, group: np.ndarray) -> np.ndarray:
        return self._realized_classifier(X, group)
