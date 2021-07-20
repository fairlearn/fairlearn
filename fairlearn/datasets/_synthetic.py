# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state


def make_sensitive_classification(feature_config=None, n_features=20, n_informative=4, random_state=None):
    """Create a synthetic dataset with a single sensitive feature: 'gender'.

    Parameters
    ----------
    feature_config : dict, default=None
        A dict whose keys are the labels for the gender feature, and whose set
        the :code:`n_samples` and :code:`class_sep` parameters for :code:`make_classification`.
        If `None`, the following dictionary is used by default::

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

    random_state : int or RandomState instance, default=None
        The random number generator seed or :class:`numpy.random.RandomState` to use.

    Returns
    -------
    (X, y, gender) : tuple of numpy.ndarray
        X : numpy.ndarray
            The generated samples.
        y : numpy.ndarray
            Labels for the binary classification.
        gender : numpy.ndarray
            The sensitive feature label.
    """
    rng = check_random_state(random_state)

    classification_kwargs = {
        'n_features': n_features,
        'n_informative': n_informative,
        'n_classes': 2,
        'random_state': rng,
    }

    if feature_config is None:
        feature_config = {
            'Man': {'n_samples': 500, 'class_sep': 1},
            'Other': {'n_samples': 500, 'class_sep': 0.5},
            'Unspecified': {'n_samples': 500, 'class_sep': 0.5},
            'Woman': {'n_samples': 500, 'class_sep': 2},
        }

    Xs, ys, genders = [], [], []

    for label, group_config in feature_config.items():
        group_config.update(classification_kwargs)
        X, y = make_classification(**group_config)
        genders.append([label] * group_config['n_samples'])
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    gender = np.concatenate(genders)

    # Now generate disparities by randomly flipping classes
    bias = {group_name: rng.uniform(0.4, 0.6) for group_name in feature_config}
    for idx in range(y.size):
        # 1/4 chance we'll overwrite this value
        if rng.uniform(0, 1) > 0.25:
            # set new value based on biased coin flip
            y[idx] = rng.uniform(0, 1) > bias[gender[idx]]

    return X, y, gender
