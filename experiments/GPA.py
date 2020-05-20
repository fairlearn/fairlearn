import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, GroupLossMoment, \
    Moment, SquareLoss
from fairlearn.metrics import group_mean_prediction, group_mean_squared_error, group_mean_overprediction
import sklearn.metrics as skm
import fairlearn.metrics as flm


class GPA:
    def __init__(self, filename, estimator, constraints):
        self.estimator = estimator
        self.constraints = constraints
        self.load_data(filename)

    def load_data(self, filename):
        names = ['gender', 'physics', 'biology', 'history', 'second_language',
                 'geography', 'literature', 'portuguese', 'math', 'chemistry', 'gpa']
        data = pd.read_csv(filename, names=names)
        y = data.gpa
        X = data.iloc[:, :-1]
        A = X.gender
        X = X.subtract(X.min(axis=0)).divide(X.max(axis=0) - X.min(axis=0))
        assert (np.all(X.max(axis=0) == 1))
        assert (np.all(X.min(axis=0) == 0))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=0.2,
                                                                                random_state=42)
        self.A_train = self.X_train.gender
        self.A_test = self.X_test.gender
        del X['gender']
        del self.X_train['gender']
        del self.X_test['gender']
        self.X_train.reset_index(inplace=True, drop=True)
        self.X_test.reset_index(inplace=True, drop=True)
        self.A_train.reset_index(inplace=True, drop=True)
        self.A_test.reset_index(inplace=True, drop=True)
        self.y_train.reset_index(inplace=True, drop=True)
        self.y_test.reset_index(inplace=True, drop=True)

    def run_baseline(self):
        lr = LinearRegression().fit(self.X_train, self.y_train)
        return lr.predict(self.X_test)

    def run_grid_search(self, grid_size, constraint_weight):
        grid_search = GridSearch(copy.deepcopy(self.estimator),
                                 constraints=copy.deepcopy(self.constraints),
                                 grid_size=grid_size,
                                 constraint_weight=constraint_weight)
        grid_search.fit(self.X_train, self.y_train, sensitive_features=self.A_train)
        return grid_search.predict(self.X_test)

    def run_exp_grad(self, eps, nu=1e-6):
        expgrad = ExponentiatedGradient(copy.deepcopy(self.estimator),
                                        constraints=copy.deepcopy(self.constraints),
                                        eps=eps,
                                        nu=nu)
        expgrad.fit(self.X_train, self.y_train, sensitive_features=self.A_train)
        return pd.Series(expgrad.predict(self.X_test), name='scores_expgrad')

    def get_mse(self, algorithm, eps=0.01, grid_size=10, constraint_weight=0.5):
        if algorithm == 'grid_search':
            y_pred = self.run_grid_search(grid_size, constraint_weight)
        elif algorithm == 'exp_grad':
            y_pred = self.run_exp_grad(eps)
        else:
            raise ValueError('Please provide valid algorithm name')
        return round(skm.mean_squared_error(self.y_test, y_pred), 4)

    def summary(self, algorithm, eps=0.01, grid_size=10, constraint_weight=0.5):
        """Provides MSE by group for a given algorithm"""
        if algorithm == 'grid_search':
            y_pred = self.run_grid_search(grid_size, constraint_weight)
        elif algorithm == 'exp_grad':
            y_pred = self.run_exp_grad(eps)
        elif algorithm == 'lin_reg':
            y_pred = self.run_baseline()
        else:
            raise ValueError('Please provide valid algorithm name')
        # mse = round(skm.mean_squared_error(self.y_test, y_pred), 4)
        mse = flm.group_mean_squared_error(self.y_test, y_pred, self.A_test)
        R = lambda x: round(x, 4)
        return R(mse['overall']), R(mse['by_group'][0.0]), R(mse['by_group'][1.0])

    @staticmethod
    def group_metric_as_df(self, name, group_metric_result):
        a = pd.Series(group_metric_result.by_group)
        a['overall'] = group_metric_result.overall
        a['disparity'] = np.abs(a.iloc[0] - a.iloc[1])
        a.rename({0: 'female', 1: 'male'}, inplace=True)
        return pd.DataFrame({name: a})

    @staticmethod
    def plot_mse_vs_eps(self, eps_values, mse):
        overall, female, male = zip(*mse)
        plt.plot(eps_values, overall, label='overall')
        plt.plot(eps_values, female, label='female')
        plt.plot(eps_values, male, label='male')
        plt.title('MSE by group for different slack in fairness constraint')
        plt.xlabel('Slack')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig('exp_grad_eps.png', dpi=300)


def main():
    filename = '~/fairlearn/experiments/data.csv'
    bgl_square_loss = GroupLossMoment(SquareLoss(-np.inf, np.inf))
    gpa = GPA(filename, LinearRegression(), bgl_square_loss)
    print('MSE by group: overall - female - male')
    print('Linear regression:', gpa.summary('lin_reg'))
    print('Grid search:', gpa.summary('grid_search'))
    print('Exponentiated gradient:', gpa.summary('exp_grad', eps=0.6))

    # Plot MSE vs Fairness trade-off
    mse = []
    eps_values = np.arange(0.1, 1.1, 0.1)
    for eps in eps_values:
        mse_result = gpa.summary('exp_grad', eps=eps)
        mse.append(mse_result)
    gpa.plot_mse_vs_eps(eps_values, mse)


if __name__ == '__main__':
    main()
