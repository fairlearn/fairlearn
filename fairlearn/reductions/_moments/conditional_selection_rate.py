# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from .moment import ClassificationMoment
from .moment import _GROUP_ID, _LABEL, _PREDICTION, _ALL, _EVENT, _SIGN, _MULTIPLIER
from fairlearn._input_validation import _MESSAGE_RATIO_NOT_IN_RANGE
from .error_rate import ErrorRate

_DIFF = "diff"


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

    def default_objective(self):
        """Return the default objective for moments of this kind."""
        return ErrorRate()

    def load_data(self, X, y, event=None, multiplier=None, **kwargs):
        """Load the specified data into this object.

        This adds a column `event` and `multiplier` to the `tags` field.

        The `multiplier` is the factor with which the signed weights are
        multiplied and correspond to g(X,A,Y,h(X)) mentioned in the paper
        `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`
        .. math::
        multiplier = g(X,A,Y,h(X)=1) - g(X,A,Y,h(X)=0)

        It defaults to 1 which implies that g(X,A,Y,h(X)) = h(X).
        """
        super().load_data(X, y, **kwargs)
        self.tags[_EVENT] = event
        if multiplier is None:
            multiplier = pd.Series(y).apply(lambda y: 1)
        self.tags[_MULTIPLIER] = multiplier
        self.prob_event = self.tags.groupby(_EVENT).size() / self.total_samples
        self.prob_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).size() / self.total_samples
        signed = pd.concat([self.prob_group_event, self.prob_group_event],
                           keys=["+", "-"],
                           names=[_SIGN, _EVENT, _GROUP_ID])
        self.index = signed.index
        self.default_objective_lambda_vec = None
        self.multiplier = self.tags.groupby([_EVENT, _GROUP_ID]).mean()
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
        pred = predictor(self.X)
        self.tags[_PREDICTION] = pred
        expect_event = self.tags.groupby(_EVENT).mean()
        expect_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).mean()
        expect_group_event[_DIFF] = expect_group_event[_PREDICTION] - expect_event[_PREDICTION]
        g_unsigned = expect_group_event[_DIFF]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=[_SIGN, _EVENT, _GROUP_ID])
        self._gamma_descr = str(expect_group_event[[_PREDICTION, _DIFF]])
        return g_signed

    # TODO: this can be further improved using the overcompleteness in group membership
    def project_lambda(self, lambda_vec):
        """Return the projected lambda values."""
        lambda_pos = lambda_vec["+"] - lambda_vec["-"]
        lambda_neg = -lambda_pos
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
        adjust = self.multiplier[_MULTIPLIER].mul(lambda_event - lambda_group_event)
        signed_weights = self.tags.apply(
            lambda row: adjust[row[_EVENT], row[_GROUP_ID]], axis=1
        )
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

    def __init__(self, ratio=1.0):
        super(DemographicParity, self).__init__()
        if ratio <= 0 or ratio > 1:
            raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
        self.ratio = ratio

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

    def __init__(self, ratio=1.0):
        super(EqualizedOdds, self).__init__()
        if ratio <= 0 or ratio > 1:
            raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
        self.ratio = ratio

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)),
                          **kwargs)


class ErrorRatio(ConditionalSelectionRate):
    r"""Implementation of Error Ratio as a moment.

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
    constraint. The value varies between 0-1. The ratio of 1 means, the
    constraint is given no relaxation and thus, the constraint tries to
    evaluate for -
    error(A=a) / total_error = 1
    """

    short_name = "ErrorRatio"

    def __init__(self, ratio=1.0):
        """Intialise with the ratio value."""
        super(ErrorRatio, self).__init__()
        if ratio <= 0 or ratio > 1:
            raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
        self.ratio = ratio

    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)),
                          multiplier=pd.Series(y).apply(lambda y: 1 - 2*y),
                          **kwargs)
