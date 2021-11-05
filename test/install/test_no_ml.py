# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pytest

from fairlearn.adversarial import AdversarialMitigationBase
from fairlearn.adversarial._constants import _IMPORT_ERROR_MESSAGE

@pytest.mark.parametrize("library", [None, 'torch', 'tensorflow'])
def test_no_library(library):
    expected_library = "torch or tensorflow" if library is None else library
    with pytest.raises(RuntimeError) as exc:
        AdversarialMitigationBase(
            library=library,
            predictor_model=[5,5],
            adversary_model=[2]
        )

        assert str(exc.value) == _IMPORT_ERROR_MESSAGE.format(expected_library)
