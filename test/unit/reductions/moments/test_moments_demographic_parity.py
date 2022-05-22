# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from fairlearn.reductions import DemographicParity
from fairlearn.reductions._moments.moment import _EVENT, _GROUP_ID, _SIGN

from .data_generator import simple_binary_threshold_data


def test_construct_and_load():
    dp = DemographicParity()
    assert dp.short_name == "DemographicParity"

    # Generate some (rigged) data
    num_samples_a0 = 10
    num_samples_a1 = 30
    num_samples = num_samples_a0 + num_samples_a1

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = 2
    a1_label = 3

    X, Y, A = simple_binary_threshold_data(
        num_samples_a0, num_samples_a1, a0_threshold, a1_threshold, a0_label, a1_label
    )

    # Load up the (rigged) data
    dp.load_data(X, Y, sensitive_features=A)
    assert dp.data_loaded
    assert dp.total_samples == num_samples_a0 + num_samples_a1

    # Examine the tags DF
    assert dp.tags["label"].equals(pd.Series(Y))
    assert dp.tags["group_id"].equals(pd.Series(A))
    assert dp.tags["event"].map(lambda x: x == "all").all()

    # Examine the index MultiIndex
    events = ["all"]
    signs = ["+", "-"]
    labels = [2, 3]
    expected_index = pd.MultiIndex.from_product(
        [signs, events, labels], names=[_SIGN, _EVENT, _GROUP_ID]
    )
    assert dp.index.equals(expected_index)

    # Examine the prob_event DF
    # There's only the 'all' event and everything belongs to it
    assert len(dp.prob_event.index) == 1
    assert dp.prob_event.loc["all"] == 1

    # Examine the prob_group_event DF
    # There's only an 'all' event but this records the fractions
    # of each label in the population
    assert len(dp.prob_group_event.index) == 2
    assert dp.prob_group_event.loc[("all", a0_label)] == num_samples_a0 / num_samples
    assert dp.prob_group_event.loc[("all", a1_label)] == num_samples_a1 / num_samples

    # Examine the neg_basis DF
    # This is obviously looking at the \lambda_{-} values and picking
    # out the one associated with the first label
    assert len(dp.neg_basis.index) == 4
    assert dp.neg_basis[0]["+", "all", a0_label] == 0
    assert dp.neg_basis[0]["+", "all", a1_label] == 0
    assert dp.neg_basis[0]["-", "all", a0_label] == 1
    assert dp.neg_basis[0]["-", "all", a1_label] == 0

    # Examine the pos_basis DF
    # This is looking at the \lambda_{+} values and picking out the
    # one associated with the first label
    assert len(dp.pos_basis.index) == 4
    assert dp.pos_basis[0]["+", "all", a0_label] == 1
    assert dp.pos_basis[0]["+", "all", a1_label] == 0
    assert dp.pos_basis[0]["-", "all", a0_label] == 0
    assert dp.pos_basis[0]["-", "all", a1_label] == 0

    # Examine the neg_basis_present DF
    assert len(dp.neg_basis_present) == 1
    assert dp.neg_basis_present[0]


def test_project_lambda_smoke_negatives():
    dp = DemographicParity()

    events = ["all"]
    signs = ["+", "-"]
    labels = ["a", "b"]
    midx = pd.MultiIndex.from_product(
        [signs, events, labels], names=[_SIGN, _EVENT, _GROUP_ID]
    )

    df = pd.DataFrame()
    # Note that the '-' indices (11 and 19) are larger
    # than the '+' indices (1 and 2)
    df = 0 + pd.Series([1, 2, 11, 19], index=midx, dtype=np.float64)

    ls = dp.project_lambda(df)

    expected = pd.DataFrame()
    expected = 0 + pd.Series([0, 0, 10, 17], index=midx, dtype=np.float64)
    assert expected.equals(ls)


def test_project_lambda_smoke_positives():
    # This is a repeat of the _negatives method but with
    # the '+' indices larger
    dp = DemographicParity()

    events = ["all"]
    signs = ["+", "-"]
    labels = ["a", "b"]
    midx = pd.MultiIndex.from_product(
        [signs, events, labels], names=[_SIGN, _EVENT, _GROUP_ID]
    )

    df = pd.DataFrame()
    # Note that the '-' indices are smaller than
    # the '+' ones
    df = 0 + pd.Series([23, 19, 5, 7], index=midx, dtype=np.float64)

    ls = dp.project_lambda(df)

    expected = pd.DataFrame()
    expected = 0 + pd.Series([18, 12, 0, 0], index=midx, dtype=np.float64)
    assert expected.equals(ls)


def test_signed_weights():
    dp = DemographicParity()
    assert dp.short_name == "DemographicParity"

    num_samples_a0 = 10
    num_samples_a1 = 30
    num_samples = num_samples_a0 + num_samples_a1

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = 0xDEAD
    a1_label = 0xBEEF

    X, Y, A = simple_binary_threshold_data(
        num_samples_a0, num_samples_a1, a0_threshold, a1_threshold, a0_label, a1_label
    )

    # Load up the (rigged) data
    dp.load_data(X, Y, sensitive_features=A)

    events = ["all"]
    signs = ["+", "-"]
    labels = [a0_label, a1_label]
    midx = pd.MultiIndex.from_product(
        [signs, events, labels], names=[_SIGN, _EVENT, _GROUP_ID]
    )

    lambda_vec = pd.Series([2000, 1000, 500, 100], index=midx, name=0)
    lambda_a0 = 2000 - 500
    lambda_a1 = 1000 - 100

    sw_a0 = (lambda_a0 + lambda_a1) - lambda_a0 * (num_samples / num_samples_a0)
    sw_a1 = (lambda_a0 + lambda_a1) - lambda_a1 * (num_samples / num_samples_a1)

    w_a0 = np.full(num_samples_a0, sw_a0)
    w_a1 = np.full(num_samples_a1, sw_a1)
    expected = np.concatenate((w_a0, w_a1), axis=None)

    signed_weights = dp.signed_weights(lambda_vec)
    assert np.array_equal(expected, signed_weights)
