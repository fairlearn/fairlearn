# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from fairlearn.reductions import EqualizedOdds
from fairlearn.reductions._moments.moment import _EVENT, _GROUP_ID, _SIGN

from .data_generator import simple_binary_threshold_data


def test_construct_and_load():
    eqo = EqualizedOdds()
    assert eqo.short_name == "EqualizedOdds"

    # Generate some rigged data
    num_samples_a0 = 10
    num_samples_a1 = 30
    num_samples = num_samples_a0 + num_samples_a1

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = 0xBEEF
    a1_label = 0xDEAD

    X, Y, A = simple_binary_threshold_data(num_samples_a0, num_samples_a1,
                                           a0_threshold, a1_threshold,
                                           a0_label, a1_label)

    # Load up the (rigged) data
    eqo.load_data(X, Y, sensitive_features=A)
    assert eqo.data_loaded
    assert eqo.total_samples == num_samples_a0 + num_samples_a1

    # Examine the tags DF
    assert eqo.tags['label'].equals(pd.Series(Y))
    assert eqo.tags['group_id'].equals(pd.Series(A))
    expected_tags_event = ['label={0}'.format(a) for a in Y]
    assert np.array_equal(expected_tags_event, eqo.tags['event'])

    # Examine the index MultiIndex
    events = ['label=False', 'label=True']
    signs = ['+', '-']
    labels = [a0_label, a1_label]
    expected_index = pd.MultiIndex.from_product(
        [signs, events, labels],
        names=[_SIGN, _EVENT, _GROUP_ID])
    assert eqo.index.equals(expected_index)

    # Examine the prob_event DF
    # There are two events - 'True' and 'False'
    assert len(eqo.prob_event.index) == 2
    assert eqo.prob_event.loc['label=False'] == 1 - sum(Y) / len(Y)
    assert eqo.prob_event.loc['label=True'] == sum(Y) / len(Y)

    # Examine the prob_group_event DF
    # There's only an 'all' event but this records the fractions
    # of each label in the population
    assert len(eqo.prob_group_event.index) == 4
    # Use the fact that our data are uniformly distributed in the range [0, 1]
    # With the current values, it appears we don't need to fiddle for off-by-one cases
    a0_below = a0_threshold * num_samples_a0
    a0_above = num_samples_a0 - a0_below
    assert eqo.prob_group_event.loc[('label=False', a0_label)] == a0_below / num_samples
    assert eqo.prob_group_event.loc[('label=True', a0_label)] == a0_above / num_samples
    a1_below = a1_threshold * num_samples_a1
    a1_above = num_samples_a1 - a1_below
    assert eqo.prob_group_event.loc[('label=False', a1_label)] == a1_below / num_samples
    assert eqo.prob_group_event.loc[('label=True', a1_label)] == a1_above / num_samples

    # Examine the neg_basis DF
    assert len(eqo.neg_basis.index) == 8
    assert eqo.neg_basis[0]['+', 'label=False', a0_label] == 0
    assert eqo.neg_basis[0]['+', 'label=False', a1_label] == 0
    assert eqo.neg_basis[0]['+', 'label=True', a0_label] == 0
    assert eqo.neg_basis[0]['+', 'label=True', a1_label] == 0
    assert eqo.neg_basis[0]['-', 'label=False', a0_label] == 1
    assert eqo.neg_basis[0]['-', 'label=False', a1_label] == 0
    assert eqo.neg_basis[0]['-', 'label=True', a0_label] == 0
    assert eqo.neg_basis[0]['-', 'label=True', a1_label] == 0

    # Examine the pos_basis DF
    # This is looking at the \lambda_{+} values and picking out the
    # one associated with the first label
    assert len(eqo.pos_basis.index) == 8
    assert eqo.pos_basis[0]['+', 'label=False', a0_label] == 1
    assert eqo.pos_basis[0]['+', 'label=False', a1_label] == 0
    assert eqo.pos_basis[0]['+', 'label=True', a0_label] == 0
    assert eqo.pos_basis[0]['+', 'label=True', a1_label] == 0
    assert eqo.pos_basis[0]['-', 'label=False', a0_label] == 0
    assert eqo.pos_basis[0]['-', 'label=False', a1_label] == 0
    assert eqo.pos_basis[0]['-', 'label=True', a0_label] == 0
    assert eqo.pos_basis[0]['-', 'label=True', a1_label] == 0

    # Examine the neg_basis_present DF
    assert len(eqo.neg_basis_present) == 2
    assert eqo.neg_basis_present[0]
    assert eqo.neg_basis_present[1]


def test_project_lambda_smoke_negatives():
    eqo = EqualizedOdds()

    events = ['label=False', 'label=True']
    signs = ['+', '-']
    labels = ['a', 'b']
    midx = pd.MultiIndex.from_product(
        [signs, events, labels],
        names=[_SIGN, _EVENT, _GROUP_ID])

    df = pd.DataFrame()
    # Note that the '-' labels are larger
    df = 0 + pd.Series([1, 2, 11, 19, 1001, 1110, 1230, 1350], index=midx, dtype=np.float64)

    ls = eqo.project_lambda(df)

    expected = pd.DataFrame()
    expected = 0 + pd.Series([0, 0, 0, 0, 1000, 1108, 1219, 1331], index=midx, dtype=np.float64)
    assert expected.equals(ls)


def test_project_lambda_smoke_positives():
    # This is a repeat of the _negatives method but with
    # the '+' indices larger
    eqo = EqualizedOdds()

    events = ['label=False', 'label=True']
    signs = ['+', '-']
    labels = ['a', 'b']
    midx = pd.MultiIndex.from_product(
        [signs, events, labels],
        names=[_SIGN, _EVENT, _GROUP_ID])

    df = pd.DataFrame()
    # Note that the '-' indices are now smaller
    df = 0 + pd.Series([200, 300, 100, 600, 4, 5, 6, 7], index=midx, dtype=np.float64)

    ls = eqo.project_lambda(df)

    expected = pd.DataFrame()
    expected = 0 + pd.Series([196, 295, 94, 593, 0, 0, 0, 0], index=midx, dtype=np.float64)
    assert expected.equals(ls)


def test_signed_weights():
    eqo = EqualizedOdds()
    assert eqo.short_name == "EqualizedOdds"

    num_samples_a0 = 10
    num_samples_a1 = 30
    num_samples = num_samples_a0 + num_samples_a1

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = 0xDEAD
    a1_label = 0xBEEF

    X, Y, A = simple_binary_threshold_data(num_samples_a0, num_samples_a1,
                                           a0_threshold, a1_threshold,
                                           a0_label, a1_label)

    # Load up the (rigged) data
    eqo.load_data(X, Y, sensitive_features=A)

    events = ['label=False', 'label=True']
    signs = ["+", "-"]
    labels = [a0_label, a1_label]
    midx = pd.MultiIndex.from_product(
        [signs, events, labels],
        names=[_SIGN, _EVENT, _GROUP_ID])

    lambda_vec = pd.Series([2000, 1000, 4000, 5000, 500, 100, 700, 900], index=midx, name=0)

    lambda_a0_F = 2000 - 500
    lambda_a0_T = 4000 - 700
    num_a0_F = int(a0_threshold * num_samples_a0)
    num_a0_T = num_samples_a0 - num_a0_F

    lambda_a1_F = 1000 - 100
    lambda_a1_T = 5000 - 900
    num_a1_F = int(a1_threshold * num_samples_a1)
    num_a1_T = num_samples_a1 - num_a1_F

    sw_a0_F = (lambda_a0_F + lambda_a1_F) / (1 - sum(Y) / len(Y)) - \
        lambda_a0_F * (num_samples / num_a0_F)
    sw_a1_F = (lambda_a0_F + lambda_a1_F) / (1 - sum(Y) / len(Y)) - \
        lambda_a1_F * (num_samples / num_a1_F)
    sw_a0_T = (lambda_a0_T + lambda_a1_T) / (sum(Y) / len(Y)) - \
        lambda_a0_T * (num_samples / num_a0_T)
    sw_a1_T = (lambda_a0_T + lambda_a1_T) / (sum(Y) / len(Y)) - \
        lambda_a1_T * (num_samples / num_a1_T)

    w_a0_F = np.full(num_a0_F, sw_a0_F)
    w_a0_T = np.full(num_a0_T, sw_a0_T)
    w_a1_F = np.full(num_a1_F, sw_a1_F)
    w_a1_T = np.full(num_a1_T, sw_a1_T)
    expected = np.concatenate((w_a0_F, w_a0_T, w_a1_F, w_a1_T), axis=None)

    signed_weights = eqo.signed_weights(lambda_vec)
    # Be bold and test for equality
    assert np.array_equal(expected, signed_weights)
