# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from fairlearn.preprocessing import OptimalTransport 

@pytest.fixture
def source_arrays():
    X=np.array([[1,1],[1,2]])
    y=np.array([[2,3],[2,5],[2,2],[5,6]])

    return X,y

def _convert_to_pandas(arr: np.array):
    cols = [str(x) for x in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)

def test_optimal_transport_numpy(source_arrays):
    X,y = source_arrays

    o = OptimalTransport(sinkhorn_regularization=0.5)
    o.fit(X,y)
    out = o.transform(X)

    assert out.shape == source_arrays[0].shape

def test_optimal_transport_pandas(source_arrays):
    X = _convert_to_pandas(source_arrays[0])
    y = _convert_to_pandas(source_arrays[1])

    o = OptimalTransport()
    o.fit(X,y)
    out = o.transform(X)

    assert out.shape == source_arrays[0].shape
    assert isinstance(out, pd.DataFrame)
    assert list(out) == list(X)
    assert all(out.index == X.index)

def test_optimal_transport_euclid_emd(source_arrays):
    X,y = source_arrays

    o = OptimalTransport(
        distance_method="euclidean",
        transport_method="emd",
    )
    o.fit(X,y)
    out = o.transform(X)

    assert out.shape == source_arrays[0].shape

def test_optimal_transport_null_y(source_arrays):
    X,_ = source_arrays

    o = OptimalTransport()

    with pytest.raises(AssertionError):
        o.fit(X,None)

def test_optimal_transport_no_fit_call(source_arrays):
    X,_ = source_arrays

    o = OptimalTransport()

    with pytest.raises(AssertionError, match=r"You must call \.fit\(\) before transforming a variable\."):
        o.transform(X)

def test_documentation_example():
    n = 1000
    is_advantaged_group = np.random.randint(2, size=n)
    predictor_1 = np.random.rand(n)
    predictor_2 = np.random.rand(n)
    label = np.random.rand(n) + (is_advantaged_group * np.random.rand(n)) # create bias

    column_names = ['is_advantaged_group','predictor_1','predictor_2','label']
    df = pd.DataFrame({k:v for k,v in zip(column_names, [is_advantaged_group, predictor_1, predictor_2, label])})

    advantaged_group = df.loc[df[column_names[0]].astype(bool), column_names[1:]]
    disadvantaged_group = df.loc[~df[column_names[0]].astype(bool), column_names[1:]]

    o = OptimalTransport()
    o.fit(disadvantaged_group, advantaged_group)
    out = o.transform(disadvantaged_group)    

    assert out.shape == disadvantaged_group.shape
