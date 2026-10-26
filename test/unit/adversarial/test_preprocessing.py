# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from numpy import asarray, issubdtype, ndarray
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from fairlearn.adversarial._preprocessor import FloatTransformer
from fairlearn.datasets import fetch_adult
from test.utils import DATA_HOME

from .helper import (
    Bin2d,
    Cat,
    Cont2d,
    MultiClass2d,
)


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
    yield [3, 2], "binary"
    yield [3, 2, 0, 1, 2, 5], "category"
    yield ["USA", "NL", "GB", "NL"], "category"
    yield [1, 2, 6, 2, 1.1], "continuous"
    yield [0.1, 2.0, 0.999999, 100000.1], "continuous"


def checker(data):
    transformer = FloatTransformer()
    transformed = transformer.fit_transform(data)
    assert isinstance(transformed, ndarray)
    assert issubdtype(transformed.dtype, float)

    original = transformer.inverse_transform(transformed)
    print(original)
    is_equal = original == data
    assert (
        isinstance(is_equal, bool)
        and is_equal
        or isinstance(is_equal, ndarray)
        and is_equal.all()
        or isinstance(is_equal, ndarray)
        and is_equal.all().all()
    )


@pytest.mark.parametrize("method_name", ["transform", "inverse_transform"])
def test_transform_before_fit_raises_not_fitted_error(method_name):
    transformer = FloatTransformer()

    with pytest.raises(NotFittedError):
        getattr(transformer, method_name)([0, 1])


@pytest.mark.parametrize("data", list(data_generator()))
@pytest.mark.parametrize("data_type", [None, asarray])
def test_data_as_datatypes(data, data_type):
    data, _, *other = data
    if data_type:
        if len(other) > 0 and data_type in other[0]:
            return
        data = data_type(data)
    checker(data)


@pytest.fixture(scope="module")
def adult_data():
    X, y = fetch_adult(return_X_y=True, data_home=DATA_HOME)
    X = X[~X.isna().any(axis=1)]
    return {
        "target": y.tolist(),
        "native-country": X["native-country"].tolist(),
        "sex": X["sex"].tolist(),
    }


@pytest.mark.openml
@pytest.mark.parametrize("column", ["target", "native-country", "sex"])
@pytest.mark.parametrize("data_type", [None, asarray])
def test_adult_data_as_datatypes(adult_data, column, data_type):
    data = adult_data[column]
    if data_type:
        data = data_type(data)
    checker(data)


@pytest.mark.parametrize(
    "transformer", ["auto", None, FunctionTransformer(None, feature_names_out="one-to-one")]
)
@pytest.mark.parametrize("data", [Bin2d, Cat, Cont2d, MultiClass2d])
def test_valid_input_data_types(data, transformer):
    """Test if the model processes the right data types"""
    prep = FloatTransformer(transformer=transformer)
    prep.fit(data)
    assert prep.n_features_in_ == data.shape[1]
    assert prep.n_features_out_ == data.shape[1]


@pytest.mark.parametrize(
    ["data", "expected_n_features_in_", "expected_n_features_out_"],
    [
        (np.array([0, 1, 1]), 1, 1),
        (np.array([[0], [1], [1]]), 1, 1),
        (np.array([[0], [1], [2]]), 1, 3),
        (np.array([[0.5], [1.1], [2]]), 1, 1),
    ],
)
def test_FloatTransformer_auto_n_features_attributes(
    data, expected_n_features_in_: int, expected_n_features_out_: int
):
    transformer = FloatTransformer("auto")
    transformer.fit(data)

    assert expected_n_features_in_ == transformer.n_features_in_
    assert expected_n_features_out_ == transformer.n_features_out_


def test_FloatTransformer_does_not_set_n_features_out_if_no_get_feature_names_out():
    transformer = FloatTransformer(FunctionTransformer(None))
    transformer.fit(np.array([0, 1]))

    assert not hasattr(transformer, "n_features_out_")


def test_FloatTransformer_passes_through_on_None():
    input_ = np.array([0, 1, 2, 0])
    transformer = FloatTransformer(None)

    output = transformer.fit_transform(input_)
    inverse = transformer.inverse_transform(output)

    np.testing.assert_array_equal(input_.astype(float).reshape(-1, 1), output)
    np.testing.assert_array_equal(input_.astype(float), inverse)


def test_FloatTransformer_multiclass_auto():
    input_ = np.array([0, 1, 2, 0])
    transformer = FloatTransformer("auto")

    expected_output = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )

    output = transformer.fit_transform(input_)
    inverse = transformer.inverse_transform(output)

    np.testing.assert_array_equal(expected_output, output)
    np.testing.assert_array_equal(input_.astype(float), inverse)


def test_FloatTransformer_custom_transformer():
    input_ = np.array([0, 1, 2, 0])
    custom_transformer = FunctionTransformer(
        func=lambda x: 2 * x, inverse_func=lambda x: x / 2, feature_names_out="one-to-one"
    )

    transformer = FloatTransformer(custom_transformer)

    expected_output = np.array([[0.0], [2.0], [4.0], [0.0]])

    output = transformer.fit_transform(input_)
    inverse = transformer.inverse_transform(output)

    np.testing.assert_array_equal(expected_output, output)
    np.testing.assert_array_equal(input_.astype(float), inverse)


def test_FloatTransformer_non_numeric_input():
    input_ = np.array(["yes", "no", "yes"])

    transformer = FloatTransformer(OneHotEncoder(sparse_output=False))

    expected_output = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    output = transformer.fit_transform(input_)
    inverse = transformer.inverse_transform(output)

    np.testing.assert_array_equal(expected_output, output)
    np.testing.assert_array_equal(input_, inverse)
