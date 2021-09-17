# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from collections import OrderedDict
from itertools import product

import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state


class SensitiveFeatureGroup:

    def __init__(self, group_assignments, **kwargs):
        """ group_assignments is a dict assigning to a group in each sensitive
        feature category. E.g.,::
            {
                'gender': 'Other',
                'age': '50-60',
            }
        """
        self.group_assignments = group_assignments
        self.classification_kwargs = kwargs

    def __repr__(self):
        return f'{self.group_assignments}={self.classification_kwargs}'


class SensitiveFeature:
    def __init__(self, name, groups):
        self.name = name
        self.groups = groups

    def __repr__(self):
        return f'{self.name}[{",".join(self.groups)}]'


class SensitiveDatasetMaker:

    def __init__(self, sensitive_features=None, random_state=None, default_group_samples=50):
        self.rng = check_random_state(random_state)
        self.sensitive_features = sensitive_features or []
        self.default_group_samples = default_group_samples
        self.init_configured_groups()

    def __repr__(self):
        features = ', '.join([str(feature) for feature in self.sensitive_features])
        return f'SensitiveDatasetMaker({features})'

    def feature_names(self):
        return [feature.name for feature in self.sensitive_features]

    def all_groups_assignments(self):
        all_groups = []
        for feature in self.sensitive_features:
            groups = [(feature.name, group_name)
                      for group_name in feature.groups]
            all_groups.append(groups)
        group_assignments = []
        for group in product(*all_groups):
            group_assignments.append(OrderedDict(
                [(feature_name, group_name) for feature_name, group_name in group]))
        return group_assignments

    def init_configured_groups(self):
        self.configured_groups = {}
        for group_assignments in self.all_groups_assignments():
            group = tuple(group_assignments.values())
            self.configured_groups[group] = SensitiveFeatureGroup(group_assignments)

    def feature_groups(self):
        for group_assignments in self.all_groups_assignments():
            yield self.configured_groups[group_assignments]

    def make_sensitive_classification(self, **kwargs):
        classification_kwargs = {
            'n_features': 20,
            'n_informative': 4,
            'n_classes': 2,
            'random_state': self.rng,
        }
        classification_kwargs.update(kwargs)

        Xs, ys = [], []
        sensitive_features = {feature_name: []
                              for feature_name in self.feature_names()}

        for group_assignments in self.all_groups_assignments():
            group = self.configured_groups[tuple(group_assignments.values())]

            group_config = group.classification_kwargs.copy()
            group_config.update(classification_kwargs)
            group_config.setdefault('weights', (self.rng.uniform(0.2, 0.8),))
            group_config.setdefault('n_samples', self.default_group_samples)

            X, y = make_classification(**group_config)
            Xs.append(X)
            ys.append(y)

            for feature in self.feature_names():
                sensitive_features[feature].extend(
                    [group_assignments[feature]] * group_config['n_samples']
                )

        X = np.concatenate(Xs)
        y = np.concatenate(ys)

        return X, y, sensitive_features
