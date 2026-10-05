# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.preprocessing import CorrelationRemover


@pytest.mark.parametrize(
    "columns",
    [
        {"feature": [10.0, 20.0, 30.0, 40.0], "sensitive": [0.0, 1.0, 0.0, 1.0]},
        {"sensitive": [0.0, 1.0, 0.0, 1.0], "renamed": [10.0, 20.0, 30.0, 40.0]},
    ],
)
def test_transform_rejects_mismatched_named_columns(constructor, columns):
    X = constructor({"sensitive": [0.0, 1.0, 0.0, 1.0], "feature": [10.0, 20.0, 30.0, 40.0]})
    remover = CorrelationRemover(sensitive_feature_ids=["sensitive"]).fit(X)
    expected = remover.transform(X)

    with pytest.raises(ValueError, match="columns and order seen during fit"):
        remover.transform(constructor(columns))

    np.testing.assert_array_equal(remover.transform(X), expected)


def test_transform_matching_named_columns(constructor):
    X = constructor({"sensitive": [0.0, 1.0, 0.0, 1.0], "feature": [10.0, 20.0, 30.0, 40.0]})
    remover = CorrelationRemover(sensitive_feature_ids=["sensitive"]).fit(X)
    np.testing.assert_allclose(remover.transform(X).ravel(), [15.0, 15.0, 35.0, 35.0])
