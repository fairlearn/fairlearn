"""Fairness metrics.

The consistency measure follows section 3 of Zemel et al. (2013),
"Learning Fair Representations": for every point we look at its k nearest
neighbours in feature space and average the distance weighted disagreement
between the point's label and its neighbours' labels. A low value means
similar individuals get similar outcomes (individual fairness).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def demographic_parity_diff(y, sensitive, unprivileged=1):
    """P(y=1 | unprivileged) - P(y=1 | privileged). 0 means parity."""
    mask_u = sensitive == unprivileged
    mask_p = ~mask_u
    return y[mask_u].mean() - y[mask_p].mean()


def disparate_impact_ratio(y, sensitive, unprivileged=1):
    """Selection rate of the unprivileged group over that of the privileged
    group. 1 means parity, <0.8 usually counts as adverse impact."""
    mask_u = sensitive == unprivileged
    mask_p = ~mask_u
    rate_u = y[mask_u].mean()
    rate_p = y[mask_p].mean()
    if rate_p == 0:
        return np.inf
    return rate_u / rate_p


def consistency(X, labels, k=10):
    """Average distance weighted label disagreement with the k nearest
    neighbours, distances normalised to [0, 1]. Lower is more consistent."""
    Z = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)

    # drop the self neighbour, then rescale distances to [0, 1] so that the
    # metric is comparable across datasets of different scales
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    dmax = dist.max()
    if dmax > 0:
        dist = dist / dmax

    labels = np.asarray(labels)
    disagreement = np.abs(labels[idx] - labels[:, None])
    return float((dist * disagreement).mean())
