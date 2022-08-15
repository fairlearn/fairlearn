# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

# This collection of tests is to verify the values computed by
# the bootstrap functionality of MetricFrame. It is based
# on the fact that if the metric is accuracy for a binary
# classification problem, then we can use the binomial
# distribution and approximate it with a gaussian.

import math

import numpy as np
import pytest

from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame

# The setup is slightly unorthodx, to allow for splitting of
# individual tests

# Overall number of samples
n_samples = 1000
# Overall fraction correct
p = 0.5
# Fraction of group 'A' as proportion of whole
f_A = 0.6
# Fraction of group 'A' which are correct
p_A = 0.8

# Everything after this point is calculated from the above

# Fraction of group 'B' as proportion of whole
f_B = 1 - f_A

# Calculate numbers of each
n_A = int(f_A * n_samples)
n_B = int((1 - f_A) * n_samples)

# Absolute number of group 'A' which are correct
n_A_correct = int(p_A * n_A)

# And for group 'B'
n_B_correct = int(n_samples * p) - n_A_correct
p_B = n_B_correct / n_B

# Absolute numbers which are incorrect
n_A_incorrect = n_A - n_A_correct
n_B_incorrect = n_B - n_B_correct


@pytest.fixture(scope="session")
def mf():
    y_true = np.ones(n_samples)
    y_pred = np.concatenate(
        (np.ones(int(n_samples * p)), np.zeros(int(n_samples * (1 - p))))
    )
    # Check for rounding errors
    assert len(y_pred) == n_samples
    assert n_A + n_B == n_samples

    # Make sure the overall metric is as expected
    assert accuracy_score(y_true, y_pred) == p

    # Sanity check
    assert n_A_correct < n_samples * p

    s_f = np.concatenate(
        (
            np.full(n_A_correct, "A"),
            np.full(n_B_correct, "B"),
            np.full(n_A_incorrect, "A"),
            np.full(n_B_incorrect, "B"),
        )
    )

    # Make sure we have avoided rounding errors
    assert len(s_f) == n_samples

    # Create the metric frame
    # The ci argument is set to have the
    # nominal value, as well as the one and two
    # sigma values
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=s_f,
        n_boot=10000,
        ci=(
            0.025,
            0.1587,
            0.5,
            0.8413,
            0.975,
        ),  # -2sigma, -sigma, nominal, sigma, 2sigma
    )

    return mf


class TestBootstrapValues:
    def test_overall(self, mf):
        assert mf.overall == p

    def test_by_group(self, mf):
        assert mf.by_group["A"] == p_A
        assert mf.by_group["B"] == p_B

    def test_quantiles_(self, mf):
        expected_quantiles = (
            0.025,
            0.1587,
            0.5,
            0.8413,
            0.975,
        )
        assert mf.quantiles_ == expected_quantiles

    def test_overall_ci(self, mf):
        mean = p * n_samples
        sigma = math.sqrt(n_samples * p * (1 - p))

        # Following corresponds to the ci argument to MetricFrame
        sigma_vals = [-2, -1, 0, 1, 2]

        for i in range(len(sigma_vals)):
            value = mean + (sigma_vals[i] * sigma)
            value_frac = value / n_samples

            assert mf.overall_ci[i][1] == pytest.approx(
                value_frac, rel=1 / math.sqrt(n_samples)
            )

    def test_by_group_ci(self, mf):
        mean_A = p_A * n_A
        sigma_A = math.sqrt(n_A * p_A * (1 - p_A))

        mean_B = p_B * n_B
        sigma_B = math.sqrt(n_B * p_B * (1 - p_B))

        # Following corresponds to the ci argument to MetricFrame
        sigma_vals = [-2, -1, 0, 1, 2]

        for i in range(len(sigma_vals)):
            v_A = mean_A + (sigma_vals[i] * sigma_A)
            vf_A = v_A / n_A

            v_B = mean_B + (sigma_vals[i] * sigma_B)
            vf_B = v_B / n_B

            # Slightly wider tolerance here.... not sure if this should be required
            assert mf.by_group_ci[i][1]["A"] == pytest.approx(
                vf_A, rel=2 / math.sqrt(n_A)
            )
            assert mf.by_group_ci[i][1]["B"] == pytest.approx(
                vf_B, rel=2 / math.sqrt(n_B)
            )

    def test_difference_between_groups(self, mf):
        # When adding or subtracting, absolute errors
        # add in quadrature
        # Only check the one sigma values here

        # Start by checking the nominal value
        assert mf.difference(method="between_groups") == pytest.approx(
            mf.difference_group_ci[2][1], rel=1 / math.sqrt(n_samples)
        )

        # Do the error calculation
        nominal_acc_AB = abs(p_A - p_B)

        sigma_A = math.sqrt(n_A * p_A * (1 - p_A))
        sigma_B = math.sqrt(n_B * p_B * (1 - p_B))
        acc_sigma_A = sigma_A / n_A
        acc_sigma_B = sigma_B / n_B

        # The actual quadrature
        acc_sigma_AB = math.sqrt(acc_sigma_A**2 + acc_sigma_B**2)

        # Sanity check
        assert nominal_acc_AB == pytest.approx(mf.difference(method="between_groups"))

        # Check the one sigma values in the output
        assert nominal_acc_AB - acc_sigma_AB == pytest.approx(
            mf.difference_group_ci[1][1], rel=1 / math.sqrt(n_samples)
        )
        assert nominal_acc_AB + acc_sigma_AB == pytest.approx(
            mf.difference_group_ci[3][1], rel=1 / math.sqrt(n_samples)
        )
