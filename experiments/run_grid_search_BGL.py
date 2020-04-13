import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, GroupLossMoment, \
    Moment, SquareLoss
from fairlearn.metrics import group_mean_prediction, group_mean_squared_error, group_mean_overprediction
import sklearn.metrics as skm
import fairlearn.metrics as flm


def loss_fct(grid_search, i):
    return grid_search.objective_weight * grid_search._objectives[i] + \
           grid_search.constraint_weight * grid_search._gammas[i].max()


def group_metric_as_df(name, group_metric_result):
    a = pd.Series(group_metric_result.by_group)
    a['overall'] = group_metric_result.overall
    return pd.DataFrame({name: a})


def main():
    names = ['gender', 'physics', 'biology', 'history', 'second_language',
             'geography', 'literature', 'portuguese', 'math', 'chemistry', 'gpa']
    data = pd.read_csv("~/fairlearn/experiments/data.csv", names=names)

    X = data.iloc[:, :-1]
    A = X.gender
    del X['gender']

    X = X.subtract(X.min(axis=0)).divide(X.max(axis=0) - X.min(axis=0))
    assert (np.all(X.max(axis=0) == 1))
    assert (np.all(X.min(axis=0) == 0))

    X['gender'] = A
    y = data.gpa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    A_train = X_train.gender
    A_test = X_test.gender
    del X_train['gender']
    del X_test['gender']

    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    A_train.reset_index(inplace=True, drop=True)
    A_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    # Least Squares
    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # # BGL Regression
    bgl_square_loss = GroupLossMoment(SquareLoss(-np.inf, np.inf))
    grid_search = GridSearch(LinearRegression(),
                             constraints=bgl_square_loss,
                             grid_size=4,
                             constraint_weight=0.5)
    grid_search.fit(X_train, y_train, sensitive_features=A_train)
    best_predict = grid_search.predict(X_test)

    # Analyze lambda
    lambda_vecs_regression = grid_search._lambda_vecs
    actual_multipliers_regression = lambda_vecs_regression.loc[1]
    actual_multipliers_regression_female = lambda_vecs_regression.loc[0]
    # print(grid_search._best_grid_index)
    losses = [loss_fct(grid_search, i) for i in range(len(grid_search._objectives))]

    mse_bgl = group_metric_as_df("MSE BGL", group_mean_squared_error(y_test, best_predict, A_test))
    mse_lr = group_metric_as_df("MSE LR", group_mean_squared_error(y_test, lr_pred, A_test))
    disp_bgl = group_metric_as_df("Disparity BGL", group_mean_prediction(y_test, best_predict, A_test))
    disp_lr = group_metric_as_df("Disparity LR", group_mean_prediction(y_test, lr_pred, A_test))
    over_pred_bgl = group_metric_as_df("overpred BGL", group_mean_overprediction(y_test, best_predict, A_test))
    over_pred_unmit = group_metric_as_df("overpred LR", group_mean_overprediction(y_test, lr_pred, A_test))

    bgl_summary_regression = pd.concat([mse_lr, mse_bgl, disp_lr, disp_bgl, over_pred_unmit, over_pred_bgl], axis=1)
    bgl_summary_regression.loc['disparity'] = (bgl_summary_regression.loc[0] - bgl_summary_regression.loc[1]).abs()
    bgl_summary_regression.loc['disparity', bgl_summary_regression.columns.str.startswith('mse')] = '-'
    bgl_summary_regression.loc['disparity', bgl_summary_regression.columns.str.startswith('overpred')] = '-'
    bgl_summary_regression.rename(index={0: "female", 1: "male"}, inplace=True)
    bgl_summary_regression.to_excel


if __name__ == '__main__':
    main()
