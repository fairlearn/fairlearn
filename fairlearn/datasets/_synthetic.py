# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state, shuffle


class SensitiveFeatureGroupConfig:

    def __init__(self, group_dict, **classification_kwargs):
        """ `group_dict` is a dict assigning to a group in each sensitive
        feature category. E.g.,::
            {
                'gender': 'Other',
                'age': '50-60',
            }
        `classification_kwargs` is the set of kwargs that will be passed on to
        make_classification.
        """
        self.group_dict = group_dict
        self.classification_kwargs = classification_kwargs

    def __repr__(self):
        return f'{self.group_dict}={self.classification_kwargs}'


class SensitiveFeature:
    """
    Parameters
    ----------
    name : str
        Name of the feature

    groups : list of str
        List of groups for the feature
    """

    def __init__(self, name, groups):
        self.name = name
        self.groups = groups

    def __repr__(self):
        return f'{self.name}[{",".join(self.groups)}]'


class SensitiveDatasetMaker:
    """ Synthetic dataset constructor with support for sensitive features.

    Parameters
    ----------
    sensitive_features : list of SensitiveFeature instances

    random_state : int or np.random.RandomState instance
        Seed or random state

    """

    def __init__(self, sensitive_features, random_state=None):
        self.rng = check_random_state(random_state)
        self.sensitive_features = sensitive_features
        self.init_configured_groups()

    def __repr__(self):
        features = ', '.join([str(feature) for feature in self.sensitive_features])
        return f'SensitiveDatasetMaker({features})'

    def all_group_dicts(self):
        """ Generate all possible group dicts in a reliable order. """
        all_groups = []
        for feature in self.sensitive_features:
            groups = [(feature.name, group_name)
                      for group_name in feature.groups]
            all_groups.append(groups)
        group_dicts = []
        for group in product(*all_groups):
            group_dicts.append(OrderedDict(
                [(feature_name, group_name) for feature_name, group_name in group]
            ))
        return group_dicts

    def init_configured_groups(self):
        """ Create a new SensitiveFeatureGroupConfig for each feature group.
        """
        self.configured_groups = {}
        for group_dict in self.all_group_dicts():
            group = tuple(group_dict.values())
            self.configured_groups[group] = SensitiveFeatureGroupConfig(group_dict)

    def make_sensitive_classification(self, n_samples_per_group=50, **kwargs):
        """ Make classification dataset with the configured sensitive features.

        Options passed to this method in :code:`**kwargs` will be forwarded to
        :code:`sklearn.datasets.make_classification`, which is responsible for
        generating the data of each feature group.

        Parameters
        ----------
        n_samples_per_group : int, default=50
            Default number of samples to generate for each group.

        Returns
        -------
        X, y, sensitive_features : Tuple(ndarray, ndarray, dict)
            A labeled dataset with sensitive_features separated from the
            non-sensitive features X.

        """
        classification_kwargs = {
            'n_features': 20,
            'n_informative': 4,
            'n_classes': 2,
            'n_samples': n_samples_per_group,
            'random_state': self.rng,
        }
        classification_kwargs.update(kwargs)

        Xs, ys = [], []
        sensitive_features = {feature.name: []
                              for feature in self.sensitive_features}

        for group_dict in self.all_group_dicts():
            group = self.configured_groups[tuple(group_dict.values())]

            group_config = classification_kwargs.copy()
            group_config.update(group.classification_kwargs)
            group_config.setdefault('weights', (self.rng.uniform(0.2, 0.8),))

            X, y = make_classification(**group_config)
            Xs.append(X)
            ys.append(y)

            for feature_name, feature_list in sensitive_features.items():
                feature_list.extend(
                    [group_dict[feature_name]] * group_config['n_samples']
                )

        X = np.concatenate(Xs)
        y = np.concatenate(ys)

        X, y, *shuffled_sensitive_features = shuffle(X, y, *sensitive_features.values())
        sensitive_features = pd.DataFrame.from_dict(
            {k: np.array(v) for k, v in zip(sensitive_features.keys(),
                                            shuffled_sensitive_features)}
        )

        return X, y, sensitive_features
