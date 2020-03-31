# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from .moment import ClassificationMoment
from .moment import _GROUP_ID, _LABEL, _PREDICTION, _ALL, _EVENT, _SIGN, \
    _POSITIVE_UTILITY, _NEGATIVE_UTILITY
from fairlearn._input_validation import _MESSAGE_RATIO_NOT_IN_RANGE, \
    _UTILITY_NOT_SPECIFIED
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

    """
    def __init__(self, ratio=1.0):
        """Intialise with the ratio value."""
        super(ConditionalSelectionRate, self).__init__()
        if ratio <= 0 or ratio > 1:
            raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
        self.ratio = ratio

    def default_objective(self):
        """Return the default objective for moments of this kind."""
        return ErrorRate()

    def load_data(self, X, y, event=None, utility=None, **kwargs):
        """Load the specified data into this object.

        This adds a column `event` to the `tags` field.

        The `utility` is a matrix which correspond to g(X,A,Y,h(X)) as mentioned
        in the paper `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`
        The `utility` defaults to h(X), ie. [1 \in R^n, 0 \in R^n]
        The `_POSITIVE_UTILITY` is G^1 and the `_NEGATIVE_UTILITY` is G^0.
        Assumes that binary class of 0/1.
        .. math::
        utility = [g(X,A,Y,h(X)=1), g(X,A,Y,h(X)=0)]

        """
        super().load_data(X, y, **kwargs)
        self.tags[_EVENT] = event
        if utility is None:
            utility = pd.Series({_POSITIVE_UTILITY: pd.Series(np.ones(y.shape)),
                                 _NEGATIVE_UTILITY: pd.Series(np.zeros(y.shape))})
        if not (_POSITIVE_UTILITY in utility and _NEGATIVE_UTILITY in utility):
            raise ValueError(_UTILITY_NOT_SPECIFIED)
        self.utility = utility
        self.prob_event = self.tags.groupby(_EVENT).size() / self.total_samples
        self.prob_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).size() / self.total_samples
        signed = pd.concat([self.prob_group_event, self.prob_group_event],
                           keys=["+", "-"],
                           names=[_SIGN, _EVENT, _GROUP_ID])
        self.index = signed.index
        self.default_objective_lambda_vec = None
        # fill in the information about the basis
        event_vals = self.tags[_EVENT].unique()
        group_vals = self.tags[_GROUP_ID].unique()
        self.pos_basis = pd.DataFrame()
        self.neg_basis = pd.DataFrame()
        self.neg_basis_present = pd.Series()
        zero_vec = pd.Series(0.0, self.index)
        i = 0
        for event_val in event_vals:
            for group in group_vals[:-1]:
                self.pos_basis[i] = 0 + zero_vec
                self.neg_basis[i] = 0 + zero_vec
                self.pos_basis[i]["+", event_val, group] = 1
                self.neg_basis[i]["-", event_val, group] = 1
                self.neg_basis_present.at[i] = True
                i += 1

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""
        utility_diff = self.utility[_POSITIVE_UTILITY] - self.utility[_NEGATIVE_UTILITY]
        pred = utility_diff * predictor(self.X) + self.utility[_NEGATIVE_UTILITY]
        self.tags[_PREDICTION] = pred
        expect_event = self.tags.groupby(_EVENT).mean()
        expect_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).mean()
        expect_group_event[_UPPER_BOUND_DIFF] = self.ratio * \
                                                expect_group_event[_PREDICTION]\
                                                - expect_event[_PREDICTION]
        expect_group_event[_LOWER_BOUND_DIFF] = - expect_group_event[_PREDICTION]\
                                                + self.ratio * expect_event[_PREDICTION]
        g_signed = pd.concat([expect_group_event[_UPPER_BOUND_DIFF],
                              expect_group_event[_LOWER_BOUND_DIFF]],
                             keys=["+", "-"],
                             names=[_SIGN, _EVENT, _GROUP_ID])
        self._gamma_descr = str(expect_group_event[[_PREDICTION, _UPPER_BOUND_DIFF, _LOWER_BOUND_DIFF]])
        return g_signed

    # TODO: this can be further improved using the overcompleteness in group membership
    def project_lambda(self, lambda_vec):
        """Return the projected lambda values."""
        lambda_pos = lambda_vec["+"] - self.ratio * lambda_vec["-"]
        lambda_neg = - self.ratio * lambda_vec["+"] + lambda_vec["-"]
        lambda_pos[lambda_pos < 0.0] = 0.0
        lambda_neg[lambda_neg < 0.0] = 0.0
        lambda_projected = pd.concat([lambda_pos, lambda_neg],
                                     keys=["+", "-"],
                                     names=[_SIGN, _EVENT, _GROUP_ID])
        return lambda_projected

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
        lambda_event = (lambda_vec["+"] - self.ratio * lambda_vec["-"]) \
            .sum(level=_EVENT) / self.prob_event
        lambda_group_event = (self.ratio * lambda_vec["+"] - lambda_vec["-"]) / \
            self.prob_group_event
        adjust = lambda_event - lambda_group_event
        signed_weights = self.tags.apply(
            lambda row: adjust[row[_EVENT], row[_GROUP_ID]], axis=1
        )
        signed_weights = (self.utility[_POSITIVE_UTILITY] -
                          self.utility[_NEGATIVE_UTILITY]).mul(signed_weights)
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
    :class:`pandas:pandas.DataFrame`
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


class EqualizedOdds(ConditionalSelectionRate):
    r"""Implementation of Equalized Odds as a moment.

    Adds conditioning on label compared to Demographic parity, i.e.

    .. math::
       P[h(X) = y' | A = a, Y = y] = P[h(X) = y' | Y = y] \; \forall a, y, y'

    This implementation of :class:`ConditionalSelectionRate` defines
    events corresponding to the unique values of the `Y` array.

    The `prob_event` :class:`pandas:pandas.DataFrame` will record the
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
    ratio <= E[abs(h(x) - y) = 1 | A = a] / E[abs(h(x) - y) = 1] <= 1/ratio\; \forall a

    This implementation of :class:`ConditionalSelectionRate` defines
    events corresponding to the unique values of the `Y` array.

    The `prob_event` :class:`pandas:pandas.DataFrame` will record the
    fraction of the samples corresponding to each unique value in
    the `Y` array.

    The `index` MultiIndex will have a number of entries equal to
    the number of unique values for the sensitive feature, multiplied by
    the number of unique values of the `Y` array, multiplied by two (for
    the Lagrange multipliers for positive and negative constraints).

    With these definitions, the :math:`signed_weights` method
    will calculate the costs according to Example 4 of
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.

    The `ratio` defines the amount of relaxation that is allowed for the
    constraint. The value varies between 0 and 1. The ratio of 1 means,
    the constraint is given no relaxation and thus, the constraint tries to
    evaluate for
    error(A=a) / total_error = 1
    """

    short_name = "ErrorRateRatio"

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y,
                          event=_ALL,
                          utility=pd.Series({_POSITIVE_UTILITY: (1-y), _NEGATIVE_UTILITY: y}),
                          **kwargs)
