from importlib.util import find_spec

import pandas as pd
import pytest
from narwhals.typing import IntoDataFrame


def pandas_constructor(obj) -> IntoDataFrame:
    return pd.DataFrame(obj)  # type: ignore[no-any-return]


# TODO: sklearn.utils.validation.check_array does not support it
# def pandas_nullable_constructor(obj) -> IntoDataFrame:
#     return pd.DataFrame(obj).convert_dtypes(dtype_backend="numpy_nullable")  # type: ignore[no-any-return]

# TODO: sklearn.utils.validation.check_array does not support it
# def pandas_pyarrow_constructor(obj) -> IntoDataFrame:
#     return pd.DataFrame(obj).convert_dtypes(dtype_backend="pyarrow")  # type: ignore[no-any-return]


def polars_eager_constructor(obj) -> IntoDataFrame:
    import polars as pl

    return pl.DataFrame(obj)


def pyarrow_table_constructor(obj) -> IntoDataFrame:
    import pyarrow as pa

    return pa.table(obj)  # type: ignore[no-any-return]


constructors = [pandas_constructor]
backends = ["pandas"]

# TODO: sklearn.utils.validation.check_array does not support these, see above
# if parse_version(pd.__version__) >= parse_version("2.0.0"):
#     constructors.extend(
#         [
#             pandas_nullable_constructor,
#             pandas_pyarrow_constructor,
#         ]
#     )

if find_spec("polars"):
    constructors.append(polars_eager_constructor)
    backends.append("polars")


if find_spec("pyarrow"):
    constructors.append(pyarrow_table_constructor)
    backends.append("pyarrow")


@pytest.fixture(params=constructors)
def constructor(request: pytest.FixtureRequest):
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=backends)
def backend(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]
