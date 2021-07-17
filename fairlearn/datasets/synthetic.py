# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
import numpy as np
from sklearn.datasets import make_classification


def make_synthetic_dataset(seed=12345):
    """Create a synthetic dataset.

    Parameters
    ----------
    seed : int, default=12345
        The random number generator seed to use.

    Returns
    -------
    (X, y, gender) : tuple of numpy.ndarray
        X : ndarray
            The generated samples.
        y : ndarray
            Labels for the binary classification.
        gender : ndarray
            The sensitive feature label.
    """
    rng = np.random.RandomState(seed=seed)

    classification_kwargs = {
        'n_features': 20,
        'n_informative': 4,
        'n_classes': 2,
        'random_state': rng,
    }

    X_women, y_women = make_classification(
        n_samples=500,
        class_sep=1,
        **classification_kwargs,
    )

    X_men, y_men = make_classification(
        n_samples=500,
        class_sep=2,
        **classification_kwargs,
    )

    X_other, y_other = make_classification(
        n_samples=500,
        class_sep=0.5,
        **classification_kwargs,
    )

    X_unspecified, y_unspecified = make_classification(
        n_samples=500,
        class_sep=0.5,
        **classification_kwargs,
    )

    X = np.r_[X_women, X_men, X_other, X_unspecified]
    y = np.r_[y_women, y_men, y_other, y_unspecified]
    gender = np.r_[
        ["Woman"] * 500,
        ["Man"] * 500,
        ["Other"] * 500,
        ["Unspecified"] * 500
    ].reshape(-1)
    return X, y, gender
