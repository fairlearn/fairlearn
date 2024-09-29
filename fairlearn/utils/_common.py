# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
from tqdm import tqdm
import torch

def _get_soft_predictions(estimator, X, predict_method):
    r"""Return soft predictions of a classifier.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn compatible estimator

    X : array-like
        The intput for which the output is desired

    predict_method : {'auto', 'predict_proba', 'decision_function', 'predict'\
            }, default='auto'

        Defines which method of the ``estimator`` is used to get the output
        values.

        - 'auto': use one of ``predict_proba``, ``decision_function``, or
          ``predict``, in that order.
        - 'predict_proba': use the second column from the output of
          `predict_proba`. It is assumed that the second column represents the
          positive outcome.
        - 'decision_function': use the raw values given by the
          `decision_function`.
        - 'predict': use the hard values reported by the `predict` method if
          estimator is a classifier, and the regression values if estimator is
          a regressor. This is equivalent to what is done in
          :footcite:`hardt2016equality`.

    Returns
    -------
    predictions : ndarray
        The output from estimator's desired predict method.

    References
    ----------
    .. footbibliography::

    """
    if predict_method == "auto":
        if hasattr(estimator, "predict_proba"):
            predict_method = "predict_proba"
        elif hasattr(estimator, "decision_function"):
            predict_method = "decision_function"
        else:
            predict_method = "predict"

    output = getattr(estimator, predict_method)(X)
    if predict_method == "predict_proba":
        return output[:, 1]
    return output

def _get_soft_predictions_batch(estimator, X, predict_method, batch_size, device):
    r"""Return soft predictions of a classifier in batches.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn compatible estimator

    X : array-like
        The intput for which the output is desired

    predict_method : {'auto', 'predict_proba', 'decision_function', 'predict'\
            }, default='auto'

        Defines which method of the ``estimator`` is used to get the output
        values.

        - 'auto': use one of ``predict_proba``, ``decision_function``, or
          ``predict``, in that order.
        - 'predict_proba': use the second column from the output of
          `predict_proba`. It is assumed that the second column represents the
          positive outcome.
        - 'decision_function': use the raw values given by the
          `decision_function`.
        - 'predict': use the hard values reported by the `predict` method if
          estimator is a classifier, and the regression values if estimator is
          a regressor. This is equivalent to what is done in
          :footcite:`hardt2016equality`.

    batch_size : int
        The number of samples in each batch
    
    device : str
        The device on which to run the computation

    Returns
    -------
    predictions : ndarray
        The output from estimator's desired predict method.

    References
    ----------
    .. footbibliography::

    """
    if predict_method == "auto":
        if hasattr(estimator, "predict_proba"):
            predict_method = "predict_proba"
        elif hasattr(estimator, "decision_function"):
            predict_method = "decision_function"
        else:
            predict_method = "predict"

    # Initialize a list to collect outputs
    output_list = []

    # Process in batches
    n_samples = X.shape[0]

    # move data to device
    x_prev_device = X.device.type
    if x_prev_device != device:
        X = X.to(device)

    estimator_prev_device = estimator.device.type
    if estimator_prev_device != device:
        estimator = estimator.to(device)

    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        batch_output = getattr(estimator, predict_method)(X_batch)

        # For predict_proba, select the probability for the positive class (usually column 1)
        if predict_method == "predict_proba":
            batch_output = batch_output[:, 1]

        output_list.append(batch_output)

    # move data back to previous device
    if x_prev_device != device:
        X = X.to(x_prev_device)
    if estimator_prev_device != device:
        estimator = estimator.to(estimator_prev_device)

    return torch.cat(output_list).cpu().numpy()
