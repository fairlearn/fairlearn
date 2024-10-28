.. _saving_loading_metricframe:

Saving and loading MetricFrame
==============================

.. currentmodule:: fairlearn.metrics

In this section, we will discuss how to save :class:`MetricFrame` in pickle
format and how to load it from a stored pickle file.

.. doctest:: saving_loading_metricframe

    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> sf_data = ['b', 'b', 'a', 'b', 'b', 'c', 'c', 'c', 'a',
    ...            'a', 'c', 'a', 'b', 'c', 'c', 'b', 'c', 'c']
    >>> from fairlearn.metrics import MetricFrame
    >>> from sklearn.metrics import accuracy_score
    >>> import pickle
    >>> metric_frame = MetricFrame(metrics=accuracy_score,
    ...                            y_true=y_true,
    ...                            y_pred=y_pred,
    ...                            sensitive_features=sf_data)
    >>> metric_frame.overall.item()
    0.444...
    >>> metric_frame.by_group
    sensitive_feature_0
    a    0.250000
    b    0.666667
    c    0.375000
    Name: accuracy_score, dtype: float64
    >>> file_name = 'metric_frame_save_load_example.pkl'
    >>> pickle.dump(metric_frame, open(file_name, 'wb'))
	>>> loaded_metric_frame = pickle.load(open(file_name, 'rb'))
    >>> loaded_metric_frame.overall.item()
    0.444...
    >>> loaded_metric_frame.by_group
    sensitive_feature_0
    a    0.250000
    b    0.666667
    c    0.375000
    Name: accuracy_score, dtype: float64
