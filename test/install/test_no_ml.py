# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pytest

from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.adversarial._constants import _IMPORT_ERROR_MESSAGE


@pytest.mark.parametrize("library", ["auto", "torch", "tensorflow"])
def test_no_library(library):
    expected_library = "torch or tensorflow" if library is None else library
    X = [[0]]
    with pytest.raises(RuntimeError) as exc:
        AdversarialFairnessClassifier(
            backend=library, predictor_model=[5, 5], adversary_model=[2]
        ).fit(X, X, sensitive_features=X)

        assert str(exc.value) == _IMPORT_ERROR_MESSAGE.format(expected_library)
