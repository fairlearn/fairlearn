# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import RocCurveDisplay

from fairlearn.metrics import MetricFrame

class RocAuc:
    """
    Provides utilties for generating AUC scores, ROC curves
    and plotting ROC curves grouped by sensitive feature. Depends on
    the Fairlearn MetricFrame class to split the input by sensitive feature.
    Uses the standard Scikit-learn modules `roc_curve`, `auc`, `roc_auc_score`
    to compute the false positive rate, true positive rate and area under the
    curve (AUC) respectively. Passing keyword arguemnts to class methods will
    pass those parameters to the underlying Scikit-learn function to allow for
    greater control.

    Reference
    ----------
    MetricFrame:
        https://fairlearn.org/v0.6.2/api_reference/fairlearn.metrics.html
    auc:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html?highlight=auc#sklearn.metrics.auc
    roc_auc_score:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
    RocCurveDisplay:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay


    Parameters
    ----------
    y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame
    The ground-truth labels for classification. (i.e. results of clf.predict())
    If labels are not either {-1, 1} or {0, 1}, then pos_label should be
    explicitly given.

    y_score : List, pandas.Series, numpy.ndarray, pandas.DataFrame
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by "decision_function" on some classifiers).

    sensitive_features : List, pandas.Series, dict of 1d arrays, numpy.ndarray,
        pandas.DataFrame
        The sensitive features which should be used to create the subgroups.
        At least one sensitive feature must be provided.
        All names (whether on pandas objects or dictionary keys) must be strings.
        We also forbid DataFrames with column names of ``None``.
        For cases where no names are provided we generate names ``sensitive_feature_[n]``.

    """
    def __init__(self,
                 y_true,
                 y_score,
                 sensitive_features):
        """
        Initiate class with required parameters to generate metrics and plots.

        Parameters
        ----------
        y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame

        y_score : List, pandas.Series, numpy.ndarray, pandas.DataFrame

        """
        self.y_true = y_true
        self.y_score = y_score
        self.sensitive_features = sensitive_features
        self.sensitive_series = self.by_group()
        self.overall_auc = roc_auc_score(self.y_true, self.y_score)
        self.auc_scores = {}
        self.__display = None

    @staticmethod
    def splitter(y_true, y_pred):
        """
        Placeholder function to enable splitting of dataframes using
        existing MetricFrame class which requires a metric.
        """
        return (y_true, y_pred)

    @staticmethod
    def plot_roc(y_true, y_score, name=None, ax=None, pos_label=1, **kwargs):
        """
        Plot Roc Curves.


        Parameters
        ----------
        y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame

        y_score : List, pandas.Series, numpy.ndarray, pandas.DataFrame

        name : The name to be used to populate the legend of the axes if
            provided.

        pos_label : The label of the positive class. When pos_label=None,
            if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1, otherwise
            an error will be raised.

        **kwargs : Optional additional key word arguments to pass to the
            Scikit-learn method `roc_curve`.

        Returns
        -------
        display : sklearn.metrics._plot.roc_curve.RocCurveDisplay object
            populated with data values as returned by the Scikit-learn module `RocCurveDisplay`. Contains the following: 'estimator_name', 'fpr',
            'tpr', 'roc_auc', 'pos_label', 'line_', 'ax_', 'figure_'.

        """
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label, **kwargs)
        auc_score = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score, estimator_name=name)
        display.plot(ax=ax)
        return display

    def plot_baseline(self, ax=None, name='Baseline'):
        """
        Plot baseline indicator. Adds an optional axes to include a
        baseline axes on the plot representing selection at random for
        ease of interpretation.
        """

        # Generate plot figure, ax if not provided
        if not ax:
            figure, ax = plt.subplots()

        # Plot baseline - 'no skill'
        # i.e. performance of classifier is equivalent to random selection
        ns_probs = [0 for n in range(len(self.y_true))]
        baseline_auc = roc_auc_score(self.y_true, ns_probs)
        ns_fpr, ns_tpr, _ = roc_curve(self.y_true, ns_probs)
        ax.plot(
            ns_fpr,
            ns_tpr,
            linestyle='--',
            color='0.8',
            label=f'{name} (AUC = {round(baseline_auc, 2)})'
               )
        return ax

    def plot_overall(self, ax=None, name='Overall', pos_label=1, **kwargs):
        """
        Plot the overall performance of the model. Adds an optional axes to plot
        the overall ROC curve for the data as a whole.

        Parameters
        ----------
        ax : A Matplotlib axes object.

        name : The name to set as the label of the axes. Default is "Overall".

        pos_label : The label of the positive class. When pos_label=None, if
            y_true is in {-1, 1} or {0, 1}, pos_label is set to 1, otherwise an
            error will be raised.

        **kwargs : Optional additional key word arguments to pass to the
            Scikit-learn method roc_curve.

        Returns
        -------
        ax : The Matplotlib axes object with data values added to the axes. If
            using the default, the axes will also include the default legend
            label (i.e. "Overall (AUC = 0.92)")

        """
        # Generate plot figure, ax if not provided
        if not ax:
            figure, ax = plt.subplots()

        # Plot overall model performance
        overall_fpr, overall_tpr, _ = roc_curve(
            self.y_true,
            self.y_score,
            pos_label=pos_label,
            **kwargs
        )
        label=f'{name} (AUC = {round(self.overall_auc, 2)})'
        ax.plot(overall_fpr, overall_tpr, label=label)
        return ax

    def by_group(self):
        """
        Splits data by sensitive feature subgroups.
        See: fairlearn.MetricFrame for more detail.

        Note: MetricFrame requires y_pred (clf.predict). However, ROC curves and
        AUC scores are generated using y_score (clf.predict_proba). This method
        substitutes y_score (type:float) for y_pred (type:int) to conform to the
        MetricFrame required params.

        Returns
        -------
        self.sensitive_series : Pandas Series object containing `y_true` and
             `y_score` indexed by sensitive feature.

        """
        mf = MetricFrame(
            metric = self.splitter,
            y_true = self.y_true,
            y_pred = self.y_score,
            sensitive_features = self.sensitive_features,
                        )
        self.sensitive_series = mf.by_group
        return self.sensitive_series

    def plot_by_group(
        self,
        sensitive_index=None,
        title = None,
        ax=None,
        include_overall=True,
        include_baseline=True,
        pos_label=1,
        **kwargs
                ):
        """
        Plots ROC curve by sensitive feature subgroup.

        To enable sub-groupings of sensitive features to be plotted separately
        and allow for more readable plots, the user may select specific features
        to plot. The object the user passes must be a Series index as returned
        by the `by_group` method. By default the legend labels will be the name
        included in the index for each sensitive feature subgrouping in the form
        of a tuple.

        The display of the plot can be customized by passing an axes object to
        the `ax` parameter.

        Parameters
        ----------

        sensitive_index : Optional Pandas Series object that contains the data
            associated with the sensitive feature indexed by subgroup. If not
            provided, default is to include all groups in the series index.

        title : Optional title to add to the resulting figure. Typically the
            name of the estimator used for classification
            (i.e. 'Logistic Regression').

        ax : A Matplotlib axes object.

        include_overall : Whether to include a plot of the ROC curve for the
            data as a whole for comparison. Default is true.

        include_baseline : Whether to include a baseline (i.e. selection at
            random) for comparison. Default is true.

        pos_label : The label of the positive class. When pos_label=None, if
            y_true is in {-1, 1} or {0, 1}, pos_label is set to 1, otherwise an
            error will be raised.

        **kwargs : Optional additional key word arguments to pass to the
            Scikit-learn method roc_curve.

        Returns
        -------
        ax : The Matplotlib axes object with data values added to the axes. If
            using the default, the axes will also include the default legend
            label (i.e. "Overall (AUC = 0.92)")

        """
        # Establish sensitive features
        if not sensitive_index:
            sensitive_index = self.sensitive_series.index

        # Generate plot figure, ax if not provided
        if not ax:
            figure, ax = plt.subplots()

        # Set plot title to name of estimator
        if title:
            ax.set_title(title)

        # Plot baseline
        if include_baseline:
            ax = self.plot_baseline(ax=ax)

        # Plot overall
        if include_overall:
            ax = self.plot_overall(ax=ax)

        # Plot ROC Curves by group
        for name in sensitive_index:
            grp_y_true = self.sensitive_series[name][0]
            grp_y_score = self.sensitive_series[name][1]
            display = self.plot_roc(
                y_true=grp_y_true,
                y_score=grp_y_score,
                name=name, #apply subgroup name to legend label
                ax=ax,
                pos_label=pos_label,
                **kwargs
            )

        return display.ax_

    def auc_by_group(self, sensitive_index=None):
        """
        Calculates AUC score by sensitive feature subgroups.
        To enable sub-groupings of sensitive features, the user may select
        specific features to plot.

        Parameters
        ----------
        sensitive_index : Optional Pandas Series object that contains the data
            associated with the  sensitive feature indexed by subgroup. If not
            provided, default is to include all groups in the series index.

        Returns
        -------
        self.auc_scores : The auc scores computed by sensitive feature groups.
            Also stored as a class object.

        """
        # Establish sensitive features
        if not sensitive_index:
            sensitive_index = self.sensitive_series.index

        for name in sensitive_index:
            grp_y_true = self.sensitive_series[name][0]
            grp_y_score = self.sensitive_series[name][1]
            grp_score = roc_auc_score(grp_y_true, grp_y_score)
            self.auc_scores[name] = grp_score
        return self.auc_scores
