# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from .moment import ClassificationMoment
from .moment import _GROUP_ID, _LABEL, _PREDICTION, _ALL, _EVENT, _SIGN
from .error_rate import ErrorRate

_DIFF = "diff"


class ConditionalSelectionRate(ClassificationMoment):
    """Generic fairness moment for selection rates.

    This serves as the base class for both :class:`DemographicParity`
    and :class:`EqualizedOdds`.
    """

    @property
    def prob_event(self):
        """Return the probability of each event.

        The events themselves are defined in the subclasses. This
        returns the fraction of rows in `X` which correspond to
        each event.
        """
        return self._prob_event

    @prob_event.setter
    def prob_event(self, value):
        self._prob_event = value

    @property
    def prob_group_event(self):
        """Return the probability of each event for each subgroup.

        The resultant DataFrame will have a number of rows equal to
        the number of unique events multiplied by the number of
        unique values for the sensitive feature.
        """
        return self._prob_group_event

    @prob_group_event.setter
    def prob_group_event(self, value):
        self._prob_group_event = value

    @property
    def index(self):
        """Return the schema for many of the computations of this object.

        This is a :class:`pandas:pandas.MultiIndex` corresponding to the rows of
        the DataFrames either required as arguments or returned by several
        of the methods of the `ConditionalSelectionRate` class.

        The :class:`pandas:pandas.MultiIndex` itself is the cartesian product of:

        - The unique events defined for the particular object
        - The unique values for the sensitive feature
        - The characters `+` and `-`, corresponding to the Lagrange multipliers
          for positive and negative violations of the constraint

        """
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def default_objective_lambda_vec(self):
        """Return the default objective lambda vector."""
        return self._default_objective_lambda_vec

    @default_objective_lambda_vec.setter
    def default_objective_lambda_vec(self, value):
        self._default_objective_lambda_vec = value

    @property
    def pos_basis(self):
        """Return a :class:`pandas:pandas.DataFrame` indexed by `index` flagging something."""
        return self._pos_basis

    @pos_basis.setter
    def pos_basis(self, value):
        self._pos_basis = value

    @property
    def neg_basis(self):
        """Return a :class:`pandas:pandas.DataFrame` indexed by `index` flagging something else."""
        return self._neg_basis

    @neg_basis.setter
    def neg_basis(self, value):
        self._neg_basis = value

    @property
    def neg_basis_present(self):
        """Return a :class:`pandas:pandas.Series` indexed by `index` flagging another thing."""
        return self._neg_basis_present

    @neg_basis_present.setter
    def neg_basis_present(self, value):
        self._neg_basis_present = value

    def default_objective(self):
        """Return the default objective for moments of this kind."""
        return ErrorRate()

    def load_data(self, X, y, event=None, **kwargs):
        """Load the specified data into this object.

        This adds a column named `event` to the :meth:`Moment.tags`
        property.
        """
        super().load_data(X, y, **kwargs)
        self.tags[_EVENT] = event
        self.prob_event = self.tags.groupby(_EVENT).size() / self.n
        self.prob_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).size() / self.n
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
        r"""Return the rebalanced Lagrange multipliers.

        For each constraint, we have two Lagrange multipliers corresponding to the
        positive and negative violations of the constraint, with both multipliers
        constrained to be positive. The 'true' Lagrange multiplier is then
        :math:`\lambda = \lambda_{+} - \lambda_{-}`. Since it makes little
        sense to the positive and negative constraints violated at the same time,
        this routine rebalances between them. For example, if we have
        :math:`\lambda_{+}=3` and :math:`\lambda_{-}=1` then this method will
        make :math:`\lambda_{+}=2` and :math:`\lambda_{-}=0`. Similarly if
        :math:`\lambda_{+}=2` and :math:`\lambda_{-}=3` then this method will
        yield :math:`\lambda_{+}=0` and :math:`\lambda_{-}=1`.

        The supplied :class:`pandas:pandas.DataFrame` must have the same
        :class:`pandas:pandas.MultiIndex` as `index` and the resultant
        :class:`pandas:pandas.DataFrame` will have that `index` as well.

        :param lambda_vec: The set of Lagrange multipliers indexed by `index`
        :type lambda_vec: :class:`pandas:pandas.DataFrame`
        """
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
        Estimator.

        :param lambda_vec: The set of Lagrange multipliers indexed by :meth:`index`
        :type lambda_vec: :class:`pandas:pandas.Series`
        """
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        adjust = lambda_signed.sum(level=_EVENT) / self.prob_event \
            - lambda_signed / self.prob_group_event
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
    (corresponding to the positive and negative Lagrange multipliers)
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
    the positive and negative Lagrange multipliers).

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
