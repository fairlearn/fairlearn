# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.adversarial._preprocessor import FloatTransformer
import pytest
from pandas import DataFrame, Series
from numpy import ndarray, asarray, issubdtype
from fairlearn.datasets import fetch_adult


def data_generator():
    """
    Generate datasets with appropriate labellings.

    Some data can not be represented by some data types, so then you
    need to provide excemption_data_types as a third tuple element. Then,
    we can skip testcases that try this.

    Yields
    ------
    (data, distribution type, excemption_data_types[optional]) : tuple
        data should be a list, distribution type either 'binary',
        'category' or 'continuous', and excemption_data_types a list
        or data structure classes.
    """
    # NOTE: we comment out mixed-type lists, as this is just weird.
    yield [0, 1, 0, 0, 1], "binary"
    yield ["hi", "person", "hi", "hi"], "binary"
    # yield [1, "hey", "hey", 1], 'binary', [asarray, Series, DataFrame]
    # yield [2.1, "hey", "hey", 2.1], 'binary', [asarray, Series, DataFrame]
    yield [3, 2], "binary"
    yield [3, 2, 0, 1, 2, 5], "category"
    yield ["USA", "NL", "GB", "NL"], "category"
    # yield [1, "hey", 4, "bye"], 'category', [asarray, Series, DataFrame]
    yield [[0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]], "category", [Series]
    yield [1, 2, 6, 2, 1.1], "continuous"
    yield [0.1, 2.0, 0.999999, 100000.1], "continuous"
    yield [[5.5, 12.2], [86.2, 81.1]], "continuous", [Series]
    yield [[0, 1], [1, 0], [0.1, 2]], "continuous", [Series]

    # Larger examples.
    X, y = fetch_adult(return_X_y=True)
    yield y.tolist(), "binary"
    non_NaN_rows = ~X.isna().any(axis=1)
    X = X[non_NaN_rows]
    yield X["native-country"].tolist(), "category"
    yield X["sex"].tolist(), "binary"
    # yield X['capital-loss'], 'continuous' # FIXME: what should this be?


def checker(data, dist_type):
    transformer = FloatTransformer()
    transformed = transformer.fit_transform(data)
    assert isinstance(transformed, ndarray)
    assert issubdtype(transformed.dtype, float)
    assert transformer.dist_type_ == dist_type

    original = transformer.inverse_transform(transformed)
    print(original)
    is_equal = original == data
    assert (
        isinstance(is_equal, bool)
        and is_equal
        or isinstance(is_equal, ndarray)
        and is_equal.all()
        or isinstance(is_equal, Series)
        and is_equal.all()
        or isinstance(is_equal, DataFrame)
        and is_equal.all().all()
    )


@pytest.mark.parametrize("data", list(data_generator()))
@pytest.mark.parametrize("data_type", [None, DataFrame, Series, asarray])
def test_data_as_datatypes(data, data_type):
    data, dist_type, *other = data
    if data_type:
        if len(other) > 0 and data_type in other[0]:
            return
        data = data_type(data)
    checker(data, dist_type)
