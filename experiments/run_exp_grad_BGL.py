import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, GroupLossMoment, \
    Moment, SquareLoss
from fairlearn.metrics import group_mean_prediction, group_mean_squared_error, group_mean_overprediction
import sklearn.metrics as skm
import fairlearn.metrics as flm


def plot_grouped_prediction_error(unmitigated_pred, mitigated_pred, grid_pred, y_true, A):
    error_df = pd.DataFrame(A)
    gender_map = {0: 'female',
                  1: 'male'}
    error_df['gender'] = error_df.gender.map(gender_map)
    error_df['lr_pred'] = unmitigated_pred - y_true
    error_df['expgrad_pred'] = mitigated_pred - y_true
    error_df['grid_pred'] = grid_pred - y_true
    fig = error_df.groupby('gender').mean().rename(columns={'lr_pred': 'LR',
                                                            'expgrad_pred': 'EXP',
                                                            'grid_pred': 'GRID'}
                                                   ).T.plot(kind='bar',
                                                            figsize=(11, 7),
                                                            title='Mean prediction error by gender',
                                                            colormap='binary',
                                                            linewidth=1,
                                                            edgecolor='black',
                                                            rot=90)
    fig.set(xlabel="Regression Algorithm", ylabel="Prediction Error")
    fig.figure.savefig('experiments/prediction_error.png', dpi=300)


def main():
    names = ['gender', 'physics', 'biology', 'history', 'second_language',
             'geography', 'literature', 'portuguese', 'math', 'chemistry', 'gpa']
    data = pd.read_csv("~/fairlearn/experiments/data.csv", names=names)

    y = data.gpa
    X = data.iloc[:, :-1]
    A = X.gender

    X = X.subtract(X.min(axis=0)).divide(X.max(axis=0) - X.min(axis=0))
    assert (np.all(X.max(axis=0) == 1))
    assert (np.all(X.min(axis=0) == 0))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
    A_train = X_train.gender
    A_test = X_test.gender
    del X['gender']
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
    print('LR MSE', skm.mean_squared_error(y_test, lr_pred))
    # print('linear regression R^2:', skm.r2_score(y_test, lr_pred))

    # BGL Regression - GridSearch
    bgl_square_loss = GroupLossMoment(SquareLoss(-np.inf, np.inf))
    grid_search = GridSearch(LinearRegression(),
                             constraints=bgl_square_loss,
                             grid_size=10,
                             constraint_weight=0.5)
    grid_search.fit(X_train, y_train, sensitive_features=A_train)
    grid_pred = grid_search.predict(X_test)
    # plt.plot(weights)
    # plt.savefig('experiments/weights_path.png')
    print('grid MSE', skm.mean_squared_error(y_test, grid_pred))
    # print('grid R^2', skm.r2_score(y_test, grid_pred))


    # Exponentiated Gradient - Regression
    bgl_square_loss = GroupLossMoment(SquareLoss(-np.inf, np.inf))
    expgrad_X = ExponentiatedGradient(
        LinearRegression(),
        constraints=bgl_square_loss,
        eps=0.01,  # TODO: try feasible eps for exponentiatied gradient
        nu=1e-6,
        is_classification_reduction=False)  # no needed anymore
    expgrad_X.fit(X_train, y_train, sensitive_features=A_train)
    expgrad_X.predict(X_test)
    exp_pred = pd.Series(expgrad_X.predict(X_test), name='scores_expgrad_X')
    print('exp-grad MSE', skm.mean_squared_error(y_test, exp_pred))
    # print('exp-grad R^2', skm.r2_score(y_test, exp_pred))

    # Visualize accuracy in bar plot
    plot_grouped_prediction_error(lr_pred, exp_pred, grid_pred, y_test, A_test)


if __name__ == '__main__':
    main()
