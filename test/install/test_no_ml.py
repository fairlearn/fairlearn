# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pytest

from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.adversarial._constants import _IMPORT_ERROR_MESSAGE


@pytest.mark.parametrize("library", ["auto", "torch", "tensorflow"])
def test_no_library(library):
    X = [[0]]
    Y = [0]
    try:
        AdversarialFairnessClassifier(
            backend=library, predictor_model=[5, 5], adversary_model=[2]
        ).fit(X, Y, sensitive_features=Y)
    except RuntimeError as error:
        print(error)
        if library == "auto":
            assert (
                str(error) == _IMPORT_ERROR_MESSAGE.format("torch or tensorflow")
                or _IMPORT_ERROR_MESSAGE.format("torch")
                or _IMPORT_ERROR_MESSAGE.format("tensorflow")
            )
        else:
            assert str(error) == _IMPORT_ERROR_MESSAGE.format(library)
