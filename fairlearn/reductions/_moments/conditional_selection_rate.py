# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from .moment import ClassificationMoment
from .moment import _GROUP_ID, _LABEL, _PREDICTION, _ALL, _EVENT, _SIGN
from fairlearn._input_validation import _MESSAGE_RATIO_NOT_IN_RANGE
from .error_rate import ErrorRate

_UPPER_BOUND_DIFF = "upper_bound_diff"
_LOWER_BOUND_DIFF = "lower_bound_diff"


class ConditionalSelectionRate(ClassificationMoment):
    """Generic fairness moment for selection rates.

    This serves as the base class for both :class:`DemographicParity`
    and :class:`EqualizedOdds`. The two are distinguished by
    the events they define, which in turn affect the
    `index` field created by :meth:`load_data()`.

    The `index` field is a :class:`pandas:pandas.MultiIndex` corresponding to the rows of
    the DataFrames either required as arguments or returned by several
    of the methods of the `ConditionalSelectionRate` class. It is the cartesian
    product of:

    - The unique events defined for the particular object
    - The unique values for the sensitive feature
    - The characters `+` and `-`, corresponding to the Lagrange multipliers
      for positive and negative violations of the constraint

    The `ratio` specifies the multiple at which error(A = a) should be compared with total_error
    and vice versa. The value of `ratio` has to be in the range (0,1] with smaller values
    corresponding to weaker constraint. The `ratio` equal to 1 corresponds to the constraint
    where error(A = a) = total_error
    """

    def __init__(self, ratio=1.0, eps=0.1):
        """Initialize with the ratio value."""
        super(ConditionalSelectionRate, self).__init__()
        if not (0 < ratio <= 1):
            raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
        self.ratio = ratio
        self.eps = eps

    def default_objective(self):
        """Return the default objective for moments of this kind."""
        return ErrorRate()

    def load_data(self, X, y, event=None, utilities=None, **kwargs):
        """Load the specified data into this object.

        This adds a column `event` to the `tags` field.

        The `utilities` is a 2-d array which correspond to g(X,A,Y,h(X)) as mentioned in the paper
        `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`. The `utilities` defaults to
        h(X), i.e. [0, 1] for each X_i. The first column is G^0 and the second is G^1.
        Assumes binary classification with labels 0/1.
        .. math::
        utilities = [g(X,A,Y,h(X)=0), g(X,A,Y,h(X)=1)]
        """
        super().load_data(X, y, **kwargs)
        self.tags[_EVENT] = event
        if utilities is None:
            utilities = np.vstack([np.zeros(y.shape, dtype=np.float64),
                                  np.ones(y.shape, dtype=np.float64)]).T
        self.utilities = utilities
        self.prob_event = self.tags.groupby(_EVENT).size() / self.total_samples
        self.prob_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).size() / self.total_samples
        signed = pd.concat([self.prob_group_event, self.prob_group_event],
                           keys=["+", "-"],
                           names=[_SIGN, _EVENT, _GROUP_ID])
        self.index = signed.index
        self.default_objective_lambda_vec = None

        # fill in the information about the basis
        event_vals = self.tags[_EVENT].dropna().unique()
        group_vals = self.tags[_GROUP_ID].unique()
        # The matrices pos_basis and neg_basis contain a lower-dimensional description of
        # constraints, which is achieved by removing some redundant constraints.
        # Considering fewer constraints is not required for correctness, but it can dramatically
        # speed up GridSearch.
        self.pos_basis = pd.DataFrame()
        self.neg_basis = pd.DataFrame()
        self.neg_basis_present = pd.Series(dtype='float64')
        zero_vec = pd.Series(0.0, self.index)
        i = 0
        for event_val in event_vals:
            # Constraints on the final group are redundant, so they are not included in the basis.
            for group in group_vals[:-1]:
                self.pos_basis[i] = 0 + zero_vec
                self.neg_basis[i] = 0 + zero_vec
                self.pos_basis[i]["+", event_val, group] = 1
                self.neg_basis[i]["-", event_val, group] = 1
                self.neg_basis_present.at[i] = True
                i += 1

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""
        utility_diff = self.utilities[:, 1] - self.utilities[:, 0]
        pred = utility_diff.T * predictor(self.X) + self.utilities[:, 0]
        self.tags[_PREDICTION] = pred
        expect_event = self.tags.groupby(_EVENT).mean()
        expect_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).mean()
        expect_group_event[_UPPER_BOUND_DIFF] = self.ratio * expect_group_event[_PREDICTION] - \
            expect_event[_PREDICTION]
        expect_group_event[_LOWER_BOUND_DIFF] = - expect_group_event[_PREDICTION] \
            + self.ratio * expect_event[_PREDICTION]
        g_signed = pd.concat([expect_group_event[_UPPER_BOUND_DIFF],
                              expect_group_event[_LOWER_BOUND_DIFF]],
                             keys=["+", "-"],
                             names=[_SIGN, _EVENT, _GROUP_ID])
        self._gamma_descr = str(expect_group_event[[_PREDICTION, _UPPER_BOUND_DIFF,
                                                    _LOWER_BOUND_DIFF]])
        return g_signed
    
    # implement bound() method ratio bound or difference bound
    def bound(self):
        return pd.Series(self.eps, index=self.index)
    # TODO: this can be further improved using the overcompleteness in group membership
    def project_lambda(self, lambda_vec):
        """Return the projected lambda values.

        i.e., returns lambda which is guaranteed to lead to the same or higher value of the
        Lagrangian compared with lambda_vec for all possible choices of the classifier, h.
        """
        if self.ratio == 1.0:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            lambda_projected = pd.concat([lambda_pos, lambda_neg],
                                         keys=["+", "-"],
                                         names=[_SIGN, _EVENT, _GROUP_ID])
            return lambda_projected
        return lambda_vec

    def signed_weights(self, lambda_vec):
        """Compute the signed weights.

        Uses the equations for :math:`C_i^0` and :math:`C_i^1` as defined
        in Section 3.2 of `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_
        in the 'best response of the Q-player' subsection to compute the
        signed weights to be applied to the data by the next call to the underlying
        estimator.

        :param lambda_vec: The vector of Lagrange multipliers indexed by `index`
        :type lambda_vec: :class:`pandas:pandas.Series`
        """
        lambda_event = (lambda_vec["+"] - self.ratio * lambda_vec["-"]).sum(level=_EVENT) / \
            self.prob_event
        lambda_group_event = (self.ratio * lambda_vec["+"] - lambda_vec["-"]) / \
            self.prob_group_event
        adjust = lambda_event - lambda_group_event
        signed_weights = self.tags.apply(
            lambda row: 0 if pd.isna(row[_EVENT]) else adjust[row[_EVENT], row[_GROUP_ID]], axis=1
        )
        utility_diff = self.utilities[:, 1] - self.utilities[:, 0]
        signed_weights = utility_diff.T * signed_weights
        return signed_weights


# Ensure that ConditionalSelectionRate shows up in correct place in documentation
# when it is used as a base class
ConditionalSelectionRate.__module__ = "fairlearn.reductions"


class DemographicParity(ConditionalSelectionRate):
    r"""Implementation of Demographic Parity as a moment.

    A classifier :math:`h(X)` satisfies DemographicParity if

    .. math::
      P[h(X) = y' | A = a] = P[h(X) = y'] \; \forall a, y'

    This implementation of :class:`ConditionalSelectionRate` defines
    a single event, `all`. Consequently, the `prob_event`
    :class:`pandas:pandas.Series`
    will only have a single entry, which will be equal to 1.
    Similarly, the `index` property will have twice as many entries
    (corresponding to the Lagrange multipliers for positive and negative constraints)
    as there are unique values for the sensitive feature.
    The :meth:`signed_weights` method will compute the costs according
    to Example 3 of
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.
    """

    short_name = "DemographicParity"

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y, event=_ALL, **kwargs)


class TruePositiveRateDifference(ConditionalSelectionRate):
    r"""Implementation of True Positive Rate Difference (Equal Opportunity Difference) as a moment.

    Adds conditioning on label `y=1` compared to Demographic parity, i.e.

    .. math::
       P[h(X) = 1 | A = a, Y = 1] = P[h(X) = 1 | Y = 1] \; \forall a

    This implementation of :class:`ConditionalSelectionRate` defines
    the event corresponding to `y=1`.

    The `prob_event` :class:`pandas:pandas.DataFrame` will record the fraction of the samples
    corresponding to `y = 1` in the `Y` array.

    The `index` MultiIndex will have a number of entries equal to the number of unique values of
    the sensitive feature, multiplied by the number of unique non-NaN values of the constructed
    `event` array, whose entries are either NaN or `label=1` (so only one unique non-NaN value),
    multiplied by two (for the Lagrange multipliers for positive and negative constraints).

    With these definitions, the :meth:`signed_weights` method will calculate the costs for `y=1` as
    they are calculated in Example 4 of `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`,
    but will use the weights equal to zero for `y=0`.
    """

    short_name = "TruePositiveRateDifference"

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        # The `where` clause is used to put `pd.nan` on all values where `y!=1`.
        super().load_data(X, y,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)).where(y == 1),
                          **kwargs)


class EqualizedOdds(ConditionalSelectionRate):
    r"""Implementation of Equalized Odds as a moment.

    Adds conditioning on label compared to Demographic parity, i.e.

    .. math::
       P[h(X) = y' | A = a, Y = y] = P[h(X) = y' | Y = y] \; \forall a, y, y'

    This implementation of :class:`ConditionalSelectionRate` defines
    events corresponding to the unique values of the `Y` array.

    The `prob_event` :class:`pandas:pandas.Series` will record the
    fraction of the samples corresponding to each unique value in
    the `Y` array.

    The `index` MultiIndex will have a number of entries equal to
    the number of unique values for the sensitive feature, multiplied by
    the number of unique values of the `Y` array, multiplied by two (for
    the Lagrange multipliers for positive and negative constraints).

    With these definitions, the :meth:`signed_weights` method
    will calculate the costs according to Example 4 of
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.
    """

    short_name = "EqualizedOdds"

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)),
                          **kwargs)


class ErrorRateRatio(ConditionalSelectionRate):
    r"""Implementation of Error Rate Ratio as a moment.

    Measures the ratio in errors per attribute by overall error.
    The 2-sided version of error ratio can be written as
    ratio <= error(A=a) / total_error <= 1/ratio
    .. math::
    ratio <= E[abs(h(x) - y)| A = a] / E[abs(h(x) - y)] <= 1/ratio\; \forall a

    This implementation of :class:`ConditionalSelectionRate` defines a single event, `all`.
    Consequently, the `prob_event` :class:`pandas:pandas.Series` will only have a single
    entry, which will be equal to 1.

    The `index` property will have twice as many entries (corresponding to the Lagrange multipliers
    for positive and negative constraints) as there are unique values for the sensitive feature.

    The :meth:`signed_weights` method will compute the costs according to Example 3 of
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.
    However, in this scenario, g = abs(h(x)-y), rather than g = h(x)
    """

    short_name = "ErrorRateRatio"

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y, event=_ALL, utilities=np.vstack([y, 1-y]).T, **kwargs)
