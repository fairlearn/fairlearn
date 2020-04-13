import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, GroupLossMoment, \
    Moment, SquareLoss
# from fairlearn.metrics import group_mean_prediction, group_mean_squared_error, group_mean_overprediction
import sklearn.metrics as skm
import fairlearn.metrics as flm


def group_metric_as_df(name, group_metric_result):
    a = pd.Series(group_metric_result.by_group)
    a['overall'] = group_metric_result.overall
    a['disparity'] = np.abs(a.iloc[0] - a.iloc[1])
    a.rename({0: 'female', 1: 'male'}, inplace=True)
    return pd.DataFrame({name: a})


def main():
    names = ['gender', 'physics', 'biology', 'history', 'second_language',
             'geography', 'literature', 'portuguese', 'math', 'chemistry', 'gpa']
    data = pd.read_csv("~/fairlearn/experiments/data.csv", names=names)

    y = data.gpa
    y = y.subtract(y.min(axis=0)).divide(y.max(axis=0) - y.min(axis=0))
    assert (np.all(y.max(axis=0) == 1))
    assert (np.all(y.min(axis=0) == 0))
    # Transform into classification
    y = y > 0.75

    X = data.iloc[:, :-1]
    A = X.gender
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,
                                                        random_state=42)
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

    # Balancing data
    balanced_index_pass0 = y_train[y_train == 1].index
    balanced_index_pass1 = y_train[y_train == 0].sample(n=balanced_index_pass0.size, random_state=0).index
    balanced_index = balanced_index_pass0.union(balanced_index_pass1)
    X_train = X_train.iloc[balanced_index, :]
    y_train = y_train.iloc[balanced_index]
    sensitive_features = A_train.iloc[balanced_index]

    R = lambda x: round(x, 4)

    # Logistic Regression
    logit = LogisticRegression().fit(X_train, y_train)
    logit_pred = logit.predict(X_test)
    print('logit accuracy:', R(sum(logit_pred == y_test) / len(y_test)))
    # dipsarity_logit = group_metric_as_df('selection_logit',
    #                                   group_mean_prediction(y_test, logit_pred, A_test))

    # Exponentiated Gradient - Equalized Odds
    expgrad_EO = ExponentiatedGradient(
        LogisticRegression(),
        constraints=EqualizedOdds(),
        eps=0.01,
        nu=1e-6)
    expgrad_EO.fit(
        X_train, y_train, sensitive_features=sensitive_features)
    EO_pred = pd.Series(expgrad_EO.predict(X_test), name='scores_expgrad_X')
    print('exp-grad EO accuracy', R(sum(EO_pred == y_test) / len(y_test)))
    # dipsarity_EO = group_metric_as_df('selection_expgrad_EO',
    #                                   group_mean_prediction(y_test, EO_pred, A_test))

    # Exponentiated Gradient - Demographic Parity
    expgrad_DP = ExponentiatedGradient(
        LogisticRegression(),
        constraints=DemographicParity(),
        eps=0.01,
        nu=1e-6)
    expgrad_DP.fit(
        X_train, y_train, sensitive_features=sensitive_features)
    DP_pred = pd.Series(expgrad_DP.predict(X_test), name='scores_expgrad_X')
    print('exp-grad DP accuracy', R(sum(DP_pred == y_test) / len(y_test)))
    # dipsarity_DP = group_metric_as_df('selection_expgrad_DP',
    #                                   group_mean_prediction(y_test, DP_pred, A_test))

    # disparity_df = pd.concat([dipsarity_logit, dipsarity_EO, dipsarity_DP], axis=1)
    # print('---------------------------------------------------------------')
    # print(disparity_df)
    # ax = disparity_df.T.plot(kind='bar', rot=30)
    # ax.legend(bbox_to_anchor=(1, 1), loc=2)
    # ax.figure.savefig('experiments/disparity.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
