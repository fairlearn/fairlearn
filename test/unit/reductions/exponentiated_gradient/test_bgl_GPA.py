import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from fairlearn.metrics import group_mean_prediction, group_mean_squared_error, group_mean_overprediction
from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds, GroupLossMoment, \
    ZeroOneLoss, SquareLoss


def main():
    names = ['gender', 'physics', 'biology', 'history', 'second_language',
             'geography', 'literature', 'portuguese', 'math', 'chemistry', 'gpa']
    data = pd.read_csv("~/fairlearn/data.csv", names=names)

    X = data.iloc[:, :-1]
    A = X.gender
    del X['gender']

    X = X.subtract(X.min(axis=0)).divide(X.max(axis=0) - X.min(axis=0))
    assert (np.all(X.max(axis=0) == 1))
    assert (np.all(X.min(axis=0) == 0))

    X['gender'] = A
    Y = data.gpa
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

    estimator = LinearRegression().fit(X, Y)
    predictor = lambda x: estimator.predict(x)
    eps = 0.05

    disparity_criterion = GroupLossMoment(SquareLoss(-np.inf, np.inf), eps)
    # disparity_criterion = GroupLossMoment(ZeroOneLoss(), eps)
    disparity_criterion.load_data(X, Y, sensitive_features=A)
    bnd = disparity_criterion.bound()
    loss_eps = disparity_criterion.gamma(predictor)
    loss = disparity_criterion.gamma(predictor)
    print(loss_eps)
    print(loss)
    print(bnd)
    assert (not np.any(loss - loss_eps))
    assert (bnd.shape == loss.shape)


if __name__ == "__main__":
    main()
