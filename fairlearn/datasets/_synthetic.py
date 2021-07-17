# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
import numpy as np
from sklearn.datasets import make_classification


def make_synthetic_dataset(classes=None, n_features=20, n_informative=4, seed=None):
    """Create a synthetic dataset with a single sensitive feature: 'gender'.

    Parameters
    ----------
    classes : dict, default=None
        A dict whose keys are the labels for the gender feature, and whose set
        the `n_samples` and `class_sep` parameters for `make_classification`.
        If `None`, the following default dict is used::

            {
                'Man': {'n_samples': 500, 'class_sep': 1},
                'Other': {'n_samples': 500, 'class_sep': 0.5},
                'Unspecified': {'n_samples': 500, 'class_sep': 0.5},
                'Woman': {'n_samples': 500, 'class_sep': 2},
            }

    n_features : int, default=20
        The total number of features.

    n_informative : int, default=4
        The number of informative features.

    seed : int, default=None
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
        'n_features': n_features,
        'n_informative': n_informative,
        'n_classes': 2,
        'random_state': rng,
    }

    if classes is None:
        classes = {
            'Man': {'n_samples': 500, 'class_sep': 1},
            'Other': {'n_samples': 500, 'class_sep': 0.5},
            'Unspecified': {'n_samples': 500, 'class_sep': 0.5},
            'Woman': {'n_samples': 500, 'class_sep': 2},
        }

    Xs, ys, genders = [], [], []

    for label, kwargs in classes.items():
        kwargs.update(classification_kwargs)
        X, y = make_classification(**kwargs)
        genders.append([label] * kwargs['n_samples'])
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    gender = np.concatenate(genders)

    return X, y, gender
