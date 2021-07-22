# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from itertools import cycle

import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state


class SensitiveFeatureGroup:

    def __init__(self, group_name, n_samples=500, class_sep=1.0):
        self.group_name = group_name
        self.n_samples = n_samples
        self.class_sep = class_sep

    def __repr__(self):
        return f'{self.group_name}={self.as_dict()}'

    def as_dict(self):
        return {'n_samples': self.n_samples, 'class_sep': self.class_sep}


class SensitiveFeature:

    def __init__(self, name):
        self.name = name
        self.groups = []

    def __repr__(self):
        groups = ', '.join([str(group) for group in self.groups])
        return f"SensitiveFeature(name='{self.name}', groups=({groups}))"

    def add_group(self, group_name, **kwargs):
        self.groups.append(SensitiveFeatureGroup(group_name, **kwargs))

    def add_groups(self, sensitive_feature_groups):
        self.groups.extend(sensitive_feature_groups)

    @property
    def total_samples(self):
        return sum(group.n_samples for group in self.groups)

    @total_samples.setter
    def total_samples(self, n_samples):
        ratio = {group: group.n_samples / self.total_samples
                 for group in self.groups}
        for group in self.groups:
            group.n_samples = int(ratio[group] * n_samples)

        missing = n_samples - self.total_samples
        # Rounding may cause us to be short
        for group in cycle(self.groups):
            if missing == 0:
                break
            group.n_samples += 1
            missing -= 1


class SensitiveDatasetMaker:

    def __init__(self, sensitive_features=None, random_state=None):
        self.rng = check_random_state(random_state)
        self.sensitive_features = sensitive_features or []

    def __repr__(self):
        features = ', '.join([str(feature) for feature in self.sensitive_features])
        return f'SensitiveDatasetMaker({features})'

    def make_sensitive_classification(self, n_samples=1000, **kwargs):
        classification_kwargs = {
            'n_features': 20,
            'n_informative': 4,
            'n_classes': 2,
            'random_state': self.rng,
        }
        classification_kwargs.update(kwargs)

        self.set_total_samples(n_samples)

        Xs, ys, sensitive_features = [], [], []

        for feature in self.sensitive_features:
            groups = []
            for group in feature.groups:
                group_config = group.as_dict()
                group_config.update(classification_kwargs)
                group_config['weights'] = (self.rng.uniform(0.2, 0.8), )
                X, y = make_classification(**group_config)
                groups.append([group.group_name] * group.n_samples)
                Xs.append(X)
                ys.append(y)
            sensitive_features.append(np.concatenate(groups))

        X = np.concatenate(Xs)
        y = np.concatenate(ys)

        return X, y, sensitive_features

    def set_total_samples(self, n_samples):
        for feature in self.sensitive_features:
            feature.total_samples = n_samples
