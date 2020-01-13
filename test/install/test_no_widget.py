# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from fairlearn import _INSTALL_WIDGET_ERROR


def test_no_matplotlib():
    with pytest.raises(RuntimeError) as exc:
        import fairlearn.widget
        assert str(exc.value) == _INSTALL_WIDGET_ERROR
