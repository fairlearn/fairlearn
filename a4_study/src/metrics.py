"""Fairness measures.

Consistency is the individual-fairness measure from section 3 of Zemel et al.
(2013): for each point, how far its label sits from the labels of its nearest
neighbours, weighted by distance. Low = similar people get similar outcomes.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def dp_diff(y, s):
    """Gap in positive-prediction rates, unprivileged minus privileged."""
    return y[s == 1].mean() - y[s == 0].mean()


def dir_ratio(y, s):
    """Ratio of the two selection rates. 1 is parity, under 0.8 is adverse
    impact by the usual four-fifths rule."""
    privileged = y[s == 0].mean()
    if privileged == 0:
        return np.inf
    return y[s == 1].mean() / privileged


def consistency(X, labels, k=10):
    Z = StandardScaler().fit_transform(X)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)

    # first neighbour is the point itself
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    # scale distances into [0, 1] so the number means the same thing for a
    # 200-row dataset as for a 48k-row one
    dist = dist / dist.max()

    labels = np.asarray(labels)
    disagree = np.abs(labels[idx] - labels[:, None])
    return float((dist * disagree).mean())
