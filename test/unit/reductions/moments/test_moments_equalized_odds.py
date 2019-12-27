# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from fairlearn.reductions import EqualizedOdds
from fairlearn.reductions._moments.moment import _EVENT, _GROUP_ID, _SIGN

from .data_generator import simple_binary_threshold_data


def test_construct_and_load():
    eqo = EqualizedOdds()
    assert eqo.short_name == "EqualizedOdds"

    num_samples_a0 = 10
    num_samples_a1 = 30
    num_samples = num_samples_a0 + num_samples_a1

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = "a0"
    a1_label = "a1"

    X, Y, A = simple_binary_threshold_data(num_samples_a0, num_samples_a1,
                                           a0_threshold, a1_threshold,
                                           a0_label, a1_label)

    # Load up the (rigged) data
    eqo.load_data(X, Y, sensitive_features=A)
    assert eqo.data_loaded
    assert eqo.n == num_samples_a0 + num_samples_a1

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
    assert eqo.neg_basis_present[0] == True


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
    df = 0 + pd.Series([1, 2, 11, 19, 1001, 1110, 1230, 1350], index=midx)

    ls = eqo.project_lambda(df)

    expected = pd.DataFrame()
    expected = 0 + pd.Series([0, 0, 0, 0, 1000, 1108, 1219, 1331], index=midx)
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
    df = 0 + pd.Series([200, 300, 100, 600, 4, 5, 6, 7], index=midx)

    ls = eqo.project_lambda(df)

    expected = pd.DataFrame()
    expected = 0 + pd.Series([196, 295, 94, 593, 0, 0, 0, 0], index=midx)
    assert expected.equals(ls)


def test_signed_weights():
    eqo = EqualizedOdds()
    assert eqo.short_name == "EqualizedOdds"

    num_samples_a0 = 10
    num_samples_a1 = 30
    num_samples = num_samples_a0 + num_samples_a1

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = "OneThing"
    a1_label = "AnotherThing"

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

    lambda_vec = pd.Series([2000, 1000, 500, 100], index=midx, name=0)
    lambda_a0 = 2000 - 500
    lambda_a1 = 1000 - 100

    sw_a0 = (lambda_a0 + lambda_a1) - lambda_a0 * (num_samples / num_samples_a0)
    sw_a1 = (lambda_a0 + lambda_a1) - lambda_a1 * (num_samples / num_samples_a1)

    w_a0 = np.full(num_samples_a0, sw_a0)
    w_a1 = np.full(num_samples_a1, sw_a1)
    expected = np.concatenate((w_a0, w_a1), axis=None)

    signed_weights = eqo.signed_weights(lambda_vec)
    assert np.array_equal(expected, signed_weights)
