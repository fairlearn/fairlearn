"""Helper functions to construct and use randomized classifiers.

TODO: this module will probably be substituted by the InterpolatedThresholder
already implemented in fairlearn.

"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.spatial import ConvexHull


class Classifier(ABC):
    @abstractmethod
    def __call__(self, X: np.ndarray, *, sensitive_features: np.ndarray = None) -> np.ndarray:
        """Return predicted class, Y, for the given input features, X.
        """
        raise NotImplementedError


class BinaryClassifier(Classifier):
    """Constructs a deterministic binary classifier, by thresholding a
    real-valued score predictor.
    """

    def __init__(
            self,
            score_predictor: callable,
            threshold: float,
        ):
        """Constructs a deterministic binary classifier from the given
        real-valued score predictor and a threshold in {0, 1}.
        """
        self.score_predictor = score_predictor
        self.threshold = threshold

    def __call__(self, X: np.ndarray, *, sensitive_features: np.ndarray = None) -> np.ndarray:
        """Computes predictions for the given samples, X.

        Parameters
        ----------
        X : np.ndarray
            The input samples, in shape (num_samples, num_features).
        sensitive_features : None, optional
            None. This argument will be ignored by this classifier as it does 
            not consider sensitive attributes.

        Returns
        -------
        y_pred_binary : np.ndarray[int]
            The predicted class for each input sample.
        """
        return (self.score_predictor(X) >= self.threshold).astype(int)


class BinaryClassifierAtROCDiagonal(Classifier):
    """A dummy classifier whose predictions have no correlation with the input
    features, but achieves whichever target FPR or TPR you want (on ROC diag.)
    """

    def __init__(
            self,
            target_fpr: float = None,
            target_tpr: float = None,
            seed: int = 42,
        ):
        err_msg = (
            f"Must provide exactly one of 'target_fpr' or 'target_tpr', "
            f"got target_fpr={target_fpr}, target_tpr={target_tpr}."
        )
        if target_fpr is not None and target_tpr is not None:
            raise ValueError(err_msg)

        # Provided FPR
        if target_fpr is not None:
            self.target_fpr = target_fpr
            self.target_tpr = target_fpr

        # Provided TPR
        elif target_tpr is not None:
            self.target_tpr = target_tpr
            self.target_fpr = target_tpr
        
        # Provided neither!
        else:
            raise ValueError(err_msg)
        
        # Initiate random number generator
        self.rng = np.random.default_rng(seed)

    def __call__(self, X: np.ndarray, *, sensitive_features: np.ndarray = None) -> np.ndarray:
        return (self.rng.random(size=len(X)) >= (1 - self.target_fpr)).astype(int)


class EnsembleGroupwiseClassifiers(Classifier):
    """Constructs a classifier from a set of group-specific classifiers.
    """

    def __init__(self, group_to_clf: dict[int | str, Callable]):
        """Constructs a classifier from a set of group-specific classifiers.

        Must be provided exactly one classifier per unique group value.

        Parameters
        ----------
        group_to_clf : dict[int | str, callable]
            A mapping of group value to the classifier that should handle 
            predictions for that specific group.
        """
        self.group_to_clf = group_to_clf

    def __call__(self, X: np.ndarray, *, sensitive_features: np.ndarray) -> np.ndarray:
        """Compute predictions for the given input samples X, given their
        sensitive attributes, `sensitive_features`.

        Parameters
        ----------
        X : np.ndarray
            Input samples, with shape (num_samples, num_features).
        group : np.ndarray, optional
            The sensitive attribute value for each input sample.

        Returns
        -------
        y_pred : np.ndarray
            The predictions, where the prediction for each sample is handed off
            to a group-specific classifier for that sample.
        """
        if len(X) != len(sensitive_features):
            raise ValueError(f"Invalid input sizes len(X) != len(group)")

        # Array to store predictions
        num_samples = len(X)
        y_pred = np.zeros(num_samples)

        # Filter to keep track of all samples that received a prediction
        cumulative_filter = np.zeros(num_samples).astype(bool)

        for group_value, group_clf in self.group_to_clf.items():
            group_filter = (sensitive_features == group_value)
            y_pred[group_filter] = group_clf(X[group_filter])
            cumulative_filter |= group_filter

        if np.sum(cumulative_filter) != num_samples:
            raise RuntimeError(
                f"Computed group-wise predictions for {np.sum(cumulative_filter)} "
                f"samples, but got {num_samples} input samples.")

        return y_pred


class RandomizedClassifier(Classifier):
    """Constructs a randomized classifier from the given  classifiers and 
    their probabilities.
    """

    def __init__(
            self,
            classifiers: list[Classifier],
            probabilities: list[float],
            seed: int = 42,
        ):
        """Constructs a randomized classifier from the given  classifiers and 
        their probabilities.
        
        This classifier will compute predictions for the whole input dataset at 
        once, which will in general be faster for larger inputs (when compared 
        to predicting each sample separately).

        Parameters
        ----------
        classifiers : list[callable]
            A list of classifiers
        probabilities : list[float]
            A list of probabilities for each given classifier, where 
            probabilities[idx] is the probability of using the prediction from 
            classifiers[idx].
        seed : int, optional
            A random seed, by default 42.

        Returns
        -------
        callable
            The corresponding randomized classifier.
        """
        if len(classifiers) != len(probabilities):
            raise ValueError(
                f"Invalid arguments: len(classifiers) != len(probabilities); "
                f"({len(classifiers)} != {len(probabilities)});")

        self.classifiers = classifiers
        self.probabilities = probabilities
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, X: np.ndarray, *, sensitive_features: np.ndarray = None) -> int:
        # Assign each sample to a classifier
        clf_idx = self.rng.choice(
            np.arange(len(self.classifiers)),       # possible choices
            size=len(X),                            # size of output array
            p=self.probabilities,                   # prob. of each choice
        )
        
        # Run predictions for all classifiers on all samples
        y_pred_choices = [clf(X) for clf in self.classifiers]
        # TODO:
        # we could actually just run the classifier for the samples that get
        # matched with it... similar to the EnsembleGroupwiseClassifiers call
        # method.
        
        return np.choose(clf_idx, y_pred_choices)


    @staticmethod
    def find_weights_given_two_points(
            point_A: np.ndarray,
            point_B: np.ndarray,
            target_point: np.ndarray,
        ):
        """Given two ROC points corresponding to existing binary classifiers,
        find the weights that result in a classifier whose ROC point is target_point.
        
        May need to interpolate the two given points with a third point corresponding
        to a random classifier (random uniform distribution with different thresholds).
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Returns a tuple of numpy arrays (Ws, Ps), such that Ws @ Ps == target_point.
            The 1st array, Ws, corresponds to the weights of each point in the 2nd array, Ps.
        """
        # Check if the target point is actually point A or B
        if all(np.isclose(point_A, target_point)):
            return np.array([1]), np.expand_dims(point_A, axis=0)

        if all(np.isclose(point_B, target_point)):
            return np.array([1]), np.expand_dims(point_B, axis=0)
        
        # If not, we'll have to triangulate the target using A and B
        point_A_fpr, point_A_tpr = point_A
        point_B_fpr, point_B_tpr = point_B
        target_fpr, target_tpr = target_point
        if not (point_A_fpr <= target_fpr <= point_B_fpr):
            raise ValueError(
                f"Invalid input. FPR should fulfill: "
                f"({point_A_fpr} point_A_FPR) <= ({target_fpr} target_fpr) <= "
                f"({point_B_fpr} point_B_fpr)")

        # Calculate weights for points A and B
        weight_A = (target_fpr - point_B_fpr) / (point_A_fpr - point_B_fpr)

        # Result of projecting target point P directly UPWARDS towards the AB line
        weights_AB = np.array([weight_A, 1 - weight_A])
        point_P_upwards = weights_AB @ np.vstack((point_A, point_B))
        if not np.isclose(point_P_upwards[0], target_fpr):
            raise RuntimeError(
                "Failed projecting target_fpr to ROC hull frontier. "
                f"Got proj. FPR={point_P_upwards[0]}; target FPR={target_fpr};")
        
        # Check if the target point lies in the AB line (and return if so)
        if all(np.isclose(point_P_upwards, target_point)):
            return weights_AB, np.vstack((point_A, point_B))

        # Result of projecting target point P directly DOWNWARDS towards the diagonal tpr==fpr
        point_P_downwards = np.array([target_fpr, target_fpr])

        # Calculate weights for P upwards and P downwards
        weight_P_upwards = (target_tpr - point_P_downwards[1]) / (point_P_upwards[1] - point_P_downwards[1])

        # Validating triangulation results
        all_points = np.vstack((point_A, point_B, point_P_downwards))
        all_weights = np.hstack((weight_P_upwards * weights_AB, 1 - weight_P_upwards))

        if not np.isclose(all_weights.sum(), 1):
            raise RuntimeError(
                f"Sum of linear interpolation weights was {all_weights.sum()}, "
                f"should be 1!")

        if not all(np.isclose(target_point, all_weights @ all_points)):
            raise RuntimeError(
                f"Triangulation of target point failed. "
                f"Target was {target_point}; got {all_weights @ all_points}.")

        return all_weights, all_points

    @staticmethod
    def construct_at_target_ROC(
            predictor: callable,
            roc_curve_data: tuple,
            target_roc_point: np.ndarray,
            seed: int = 42,
        ) -> "RandomizedClassifier":
        """Constructs a randomized classifier in the interior of the
        convex hull of the classifier's ROC curve, at a given target
        ROC point.
        
        Parameters
        ----------
        predictor : callable
            A predictor that outputs real-valued scores in range [0; 1].
        roc_curve_data : tuple[np.array...]
            The ROC curve of the given classifier, as a tuple of
            (FPR values; TPR values; threshold values).
        target_roc_point : np.ndarray
            The target ROC point in (FPR, TPR).
        
        Returns
        -------
        rand_clf : callable
            A (randomized) binary classifier whose expected FPR and TPR
            corresponds to the given target ROC point.
        """
        # Unpack useful constants
        target_fpr, target_tpr = target_roc_point
        fpr, tpr, thrs = roc_curve_data

        # Check if we have more than two ROC points
        # (3 minimum to compute convex hull)
        if len(fpr) <= 1:
            raise ValueError(
                f"Invalid ROC curve data (only has one point): "
                f"fpr:{fpr}; tpr:{tpr}.")

        if len(fpr) == 2:
            logging.warning(f"Got ROC data with only 2 points: producing a random classifier...")
            if not np.isclose(target_roc_point[0], target_roc_point[1]):
                logging.error(
                    f"Invalid target ROC point ({target_roc_point}) is not in "
                    "diagonal ROC line, but a random-classifier ROC was provided.")

            return BinaryClassifierAtROCDiagonal(target_fpr=target_roc_point[0])

        # Compute hull of ROC curve
        roc_curve_points = np.stack((fpr, tpr), axis=1)
        hull = ConvexHull(roc_curve_points)

        # Filter out ROC points in the interior of the convex hull and other suboptimal points
        points_above_diagonal = np.argwhere(tpr >= fpr).ravel()
        useful_points_idx = np.array(sorted(set(hull.vertices) & set(points_above_diagonal)))

        fpr = fpr[useful_points_idx]
        tpr = tpr[useful_points_idx]
        thrs = thrs[useful_points_idx]

        # Find points A and B to construct the randomized classifier from
        # > point A is the last point with FPR smaller or equal to the target
        point_A_idx = 0
        if target_fpr > 0:
            point_A_idx = max(np.argwhere(fpr <= target_fpr).ravel())
        point_A_roc = roc_curve_points[useful_points_idx][point_A_idx]

        # > point B is the first point with FPR larger than the target
        point_B_idx = min(point_A_idx + 1, len(thrs) - 1)
        point_B_roc = roc_curve_points[useful_points_idx][point_B_idx]

        weights, points = RandomizedClassifier.find_weights_given_two_points(
            point_A=point_A_roc,
            point_B=point_B_roc,
            target_point=target_roc_point,
        )

        if max(weights) > 1:
            logging.error(f"Got triangulation weights over 100%: w={weights};")

        # Instantiate classifiers for points A and B
        clf_a = BinaryClassifier(predictor, threshold=thrs[point_A_idx])
        clf_b = BinaryClassifier(predictor, threshold=thrs[point_B_idx])

        # Check if most of the probability mass is on a single classifier
        if np.isclose(max(weights), 1.0):
            if all(np.isclose(target_roc_point, point_A_roc)):
                return clf_a

            elif all(np.isclose(target_roc_point, point_B_roc)):
                return clf_b
            
            else:
                # differences from target point to A or B are significant enough
                # to warrant triangulating between multiple points
                pass

        # If only one point returned, then that point should have weight==1.0
        # (hence, should've been caught by the previous if statement)
        if len(weights) == 1:
            raise RuntimeError("Invalid triangulation.")
        
        # If there are two points, return a randomized classifier between the two
        elif len(weights) == 2:
            return RandomizedClassifier(
                classifiers=[clf_a, clf_b],
                probabilities=weights,
                seed=seed,
            )

        # If it's in the interior of the ROC curve, requires instantiating a randomized classifier at the diagonal
        elif len(weights) == 3:
            fpr_rand, tpr_rand = points[2]
            if not np.isclose(fpr_rand, tpr_rand):
                raise RuntimeError(
                    f"Triangulation point at ROC diagonal has FPR != TPR "
                    f"({fpr_rand} != {tpr_rand}); ")

            # >>> BUG this would be better but for some reason it doesn't work!
            # rng = np.random.default_rng(42)
            # clf_rand = lambda X: (rng.random(size=len(X)) >= (1 - fpr_rand)).astype(int)
            # # or...
            # clf_rand = BinaryClassifierAtROCDiagonal(target_fpr=fpr_rand)
            # <<<
            clf_rand = lambda X: (np.random.random(size=len(X)) >= (1 - fpr_rand)).astype(int)

            return RandomizedClassifier(
                classifiers=[clf_a, clf_b, clf_rand],
                probabilities=weights,
                seed=seed)
        
        else:
            raise RuntimeError(
                f"Invalid triangulation of classifiers; "
                f"weights: {weights}; points: {points};")

    @staticmethod
    def find_points_for_target_ROC(roc_curve_data, target_roc_point):
        """Retrieves a set of realizable points (and respective weights) in the
        provided ROC curve that can be used to realize any target ROC in the
        interior of the ROC curve.

        NOTE: this method is a bit redundant -- has functionality in common with
        RandomizedClassifier.construct_at_target_ROC()
        """
        # Unpack useful constants
        target_fpr, target_tpr = target_roc_point
        fpr, tpr, thrs = roc_curve_data

        # Compute hull of ROC curve
        roc_curve_points = np.stack((fpr, tpr), axis=1)
        hull = ConvexHull(roc_curve_points)

        # Filter out ROC points in the interior of the convex hull and other suboptimal points
        points_above_diagonal = np.argwhere(tpr >= fpr).ravel()
        useful_points_idx = np.array(sorted(set(hull.vertices) & set(points_above_diagonal)))

        fpr = fpr[useful_points_idx]
        tpr = tpr[useful_points_idx]
        thrs = thrs[useful_points_idx]

        # Find points A and B to construct the randomized classifier from
        # > point A is the last point with FPR smaller or equal to the target
        point_A_idx = max(np.argwhere(fpr <= target_fpr).ravel())
        # > point B is the first point with FPR larger than the target
        point_B_idx = point_A_idx + 1

        weights, points = RandomizedClassifier.find_weights_given_two_points(
            point_A=roc_curve_points[useful_points_idx][point_A_idx],
            point_B=roc_curve_points[useful_points_idx][point_B_idx],
            target_point=target_roc_point,
        )

        return weights, points
