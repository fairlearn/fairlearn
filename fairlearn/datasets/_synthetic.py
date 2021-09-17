# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state


class SensitiveFeatureGroup:

    def __init__(self, group_dict, **classification_kwargs):
        """ group_dict is a dict assigning to a group in each sensitive
        feature category. E.g.,::
            {
                'gender': 'Other',
                'age': '50-60',
            }
        """
        self.group_dict = group_dict
        self.classification_kwargs = classification_kwargs

    def __repr__(self):
        return f'{self.group_dict}={self.classification_kwargs}'


class SensitiveFeature:
    def __init__(self, name, groups):
        self.name = name
        self.groups = groups

    def __repr__(self):
        return f'{self.name}[{",".join(self.groups)}]'


class SensitiveDatasetMaker:

    def __init__(self, sensitive_features=None, random_state=None):
        self.rng = check_random_state(random_state)
        self.sensitive_features = sensitive_features or []
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
        """ Create a new SensitiveFeatureGroup for each feature group. """
        self.configured_groups = {}
        for group_dict in self.all_group_dicts():
            group = tuple(group_dict.values())
            self.configured_groups[group] = SensitiveFeatureGroup(group_dict)

    def make_sensitive_classification(self, **kwargs):
        classification_kwargs = {
            'n_features': 20,
            'n_informative': 4,
            'n_classes': 2,
            'n_samples': 50,
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

        sensitive_features = pd.DataFrame.from_dict(
            {k: np.array(v) for k, v in sensitive_features.items()})

        return X, y, sensitive_features
