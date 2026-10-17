# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
"""Fixtures scoped to the ``fairlearn.utils`` input-validation tests.

The shared ``constructor`` fixture in ``test/conftest.py`` also parametrizes over
pandas DataFrames backed by the ``numpy_nullable`` and ``pyarrow`` extension
dtypes. The utilities exercised here funnel their inputs through
:func:`sklearn.utils.validation.check_array`, which materializes the input as a
NumPy array and does not support those pandas extension-dtype backends (for
example, converting a ``string[pyarrow]`` column raises
``ValueError: could not convert string to float``). We therefore override
``constructor`` locally to cover only the backends these utilities support:
regular pandas, polars, and native pyarrow tables.
"""

from importlib.util import find_spec

import pytest

from test.conftest import (
    pandas_constructor,
    polars_eager_constructor,
    pyarrow_table_constructor,
)

constructors = [pandas_constructor]

if find_spec("polars"):
    constructors.append(polars_eager_constructor)

if find_spec("pyarrow"):
    constructors.append(pyarrow_table_constructor)


@pytest.fixture(params=constructors)
def constructor(request: pytest.FixtureRequest):
    return request.param  # type: ignore[no-any-return]
