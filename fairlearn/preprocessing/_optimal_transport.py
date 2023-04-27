# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import ot
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Union

class OptimalTransport(BaseEstimator, TransformerMixin):
    r"""
    An optimal transport adjustment for the relevant joint probability 
    distributions of an "advantaged" and "disadvanted" group. The 
    adjustment is obtained by applying optimal transport on the 
    disadvantaged group to map its distribution to that of the advantaged
    group. 

    For distance calculations, we wrap the `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy-spatial-distance-cdist>`_
    method, so all methods supported by cdist are supported here as well.

    For optimal transport, we leverage the `POT <https://pythonot.github.io/>`_ library and support 
    the `emd <https://pythonot.github.io/all.html#ot.emd`_ and `sinkhorn <https://pythonot.github.io/all.html#ot.sinkhorn>_ method.


    Parameters
    ----------
    distance_method: str, default="seuclidean"
        Input to the code:`scipy.spatial.distance.cdist` code:`metric` parameter. 

    transport_method: str, default="sinkhorn"    
        Optimal transport method. Options are:

            'emd'
                earth movers distance method
            
            'sinkhorn'
                sinkhorn method

    sinkhorn_regularization: int, default=1
        Sinkhorn regularization parameter. See the `POT docs <https://pythonot.github.io/all.html#ot.sinkhorn>`_ for more.

    Notes
    -----
    The procedure is based on the algorithm of
    `Berk et al. (2022) <https://arxiv.org/abs/1610.02413`_ [1]_.

    References
    ----------
    .. [1] Richard A. Berk, Arun Kumar Kuchibhotla, Eric Tchetgen Tchetgen, 
       "Improving Fairness in Criminal Justice Algorithmic Risk Assessments 
       Using Optimal Transport and Conformal Prediction Sets," arXiv.org,
       09-Aug-2022.
       [Online]. Available: https://arxiv.org/abs/2111.09211

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from fairlearn.preprocessing import OptimalTransport 
    >>> n = 1000
    >>> is_advantaged_group = np.random.randint(2, size=n)
    >>> predictor_1 = np.random.rand(n)
    >>> predictor_2 = np.random.rand(n)
    >>> label = np.random.rand(n) + (is_advantaged_group * np.random.rand(n)) # create bias
    >>> column_names = ['is_advantaged_group','predictor_1','predictor_2','label']
    >>> df = pd.DataFrame({k:v for k,v in zip(column_names, [is_advantaged_group, predictor_1, predictor_2, label])})
    >>> advantaged_group = df.loc[df[column_names[0]].astype(bool), column_names[1:]]
    >>> disadvantaged_group = df.loc[~df[column_names[0]].astype(bool), column_names[1:]]
    >>> o = OptimalTransport()
    >>> o.fit(disadvantaged_group, advantaged_group)
    >>> transformed = o.transform(disadvantaged_group)    
    """
    
    def __init__(
        self, 
        *,
        distance_method="seuclidean", 
        transport_method="sinkhorn", 
        sinkhorn_regularization=1
    ):
        self.distance_method = distance_method
        self.transport_method = transport_method
        self.sinkhorn_regularization = sinkhorn_regularization

    def fit(self, X, y=None):
        """
        Learn the transpot plan that maps X to y. Supported distance metrics 
        include those in the scipy.spatial.distance.cdist metrics parameter.
        Supported optimal transport methods include sinkhorn and earth movers
        distance. See the class docs for more information.
        
        This function does not return variables; it simply creates the gamma 
        (transport plan) and y (target distribution) class variables.


        Parameters
        ----------
            X : np.array, pd.DataFrame 
                source distribution that will learn the mapping to y. This should 
                represent the "disadvantaged" class.
            y : np.array, pd.DataFrame 
                target distribution that x will map onto. This should represent 
                the "advantaged" class.
        
        """

        assert isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame), "X must be a np.ndarray or pd.DataFrame."
        assert isinstance(y, np.ndarray) or isinstance(y, pd.DataFrame), "y must be a np.ndarray or pd.DataFrame."
        assert y is not None, "y must be specified."

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        n_X, n_y = len(X), len(y)
        marginal_x = np.ones(n_X) / n_X
        marginal_y = np.ones(n_y) / n_y

        dist = ot.dist(X, y, metric = self.distance_method) 

        if self.transport_method == "sinkhorn":
            gamma = ot.sinkhorn(marginal_x, marginal_y, dist, self.sinkhorn_regularization)
        elif self.transport_method == "emd":
            gamma = ot.emd(marginal_x, marginal_y, dist)

        self.gamma = gamma
        self.y = y

    def transform(self, X):
        """
        Convert X's distribution to that of the target variable (y) specified in the 
        .fit() function.

        Parameters
        ----------
            X : np.array, pd.DataFrame 
                source distribution that will transformed to the target distribution.
        
        Returns
        -------
        numpy.ndarray, pd.DataFrame
            a transformed version of X
        """
        assert hasattr(self, 'gamma') and hasattr(self, 'y'), "You must call .fit() before transforming a variable."
        assert isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame), "X must be a np.ndarray or pd.DataFrame."

        was_pandas = False
        if isinstance(X, pd.DataFrame):
            index = X.index
            columns = list(X)
            was_pandas = True

        source = X
        transported = np.zeros(source.shape)
        for i in range(len(source)):
            transported[i,:] = np.sum(self.gamma[i,:]*self.y.T, axis=1) / np.sum(self.gamma[i,:]) # Compute average Yi

        if was_pandas:
            transported = pd.DataFrame(transported, columns = columns)
            transported.index = index

        return transported 
