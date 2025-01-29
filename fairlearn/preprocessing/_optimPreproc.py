import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ._optimPreproc_helper import DTools


class OptimizedPreprocessor(TransformerMixin, BaseEstimator):
    """Optimized preprocessing is a preprocessing technique that learns a
    probabilistic transformation that edits the features and labels in the data
    with group fairness, individual distortion, and data fidelity constraints
    and objectives [1]_.

    References:
        .. [1] F. P. Calmon, D. Wei, B. Vinzamuri, K. Natesan Ramamurthy, and
           K. R. Varshney. "Optimized Pre-Processing for Discrimination
           Prevention." Conference on Neural Information Processing Systems,
           2017.

    Based on code available at: https://github.com/fair-preprocessing/nips2017
    """

    def __init__(
        self,
        sensitive_feature_ids,
        distortion_function,
        epsilon=0.05,
        clist=[0.99, 1.99, 2.99],
        dlist=[0.1, 0.05, 0],
        verbose=False,
        seed=None,
    ) -> None:
        super().__init__()

        self.seed = seed
        self.verbose = verbose
        self.sensitive_feature_ids = sensitive_feature_ids
        self.distortion_function = distortion_function
        self.epsilon = epsilon
        self.clist = clist
        self.dlist = dlist

    def createDataframe(self, X, y, prefix="column"):
        """Convert input data into pandas DataFrames with appropriate column names.

        Args:
            X: Input features as numpy array or pandas DataFrame
            y: Target values
            prefix: Prefix for auto-generated column names when X is a numpy array

        Returns:
            tuple: (features_df, target_df, sensitive_df) containing:
                - features_df: DataFrame of non-sensitive features
                - target_df: DataFrame of target values
                - sensitive_df: DataFrame of sensitive features
        """
        n_cols = X.shape[1]
        if isinstance(X, pd.DataFrame):
            sensitive_df = X[self.sensitive_feature_ids]
            X_df = X.drop(columns=self.sensitive_feature_ids)
            return X_df, y, sensitive_df
        elif isinstance(X, np.ndarray):
            column_names = [f"{prefix}_{i}" for i in range(1, n_cols + 1)]
            sensitive_columns = []
            for ids in self.sensitive_feature_ids:
                sensitive_columns.append(column_names[ids])
            X_df = pd.DataFrame(X, columns=column_names)
            sensitive_df = X_df[sensitive_columns]
            X_df = X_df.drop(columns=sensitive_columns)
            y_df = pd.DataFrame(y, columns=[f"{prefix}_{n_cols+1}"])
            return X_df, y_df, sensitive_df

    def fit(self, X, y):
        """Fit the preprocessor to the input data.

        Args:
            X: Input features
            y: Target values

        Returns:
            self: The fitted preprocessor
        """
        X_df, y_df, sensitive_df = self.createDataframe(X, y)
        X_features = X_df.columns.to_list()
        y_features = y_df.columns.to_list()
        sensitive_names = sensitive_df.columns.to_list()
        all_features = X_features + sensitive_names + y_features

        combined_df = pd.concat([X_df, sensitive_df], axis=1)
        combined_df = pd.concat([combined_df, y_df], axis=1)

        self.optimizer = DTools(df=combined_df, features=all_features)
        self.optimizer.setFeatures(D=sensitive_names, X=X_features, Y=y_features)
        self.optimizer.set_distortion(self.distortion_function, self.clist)

        self.optimizer.optimize(
            epsilon=self.epsilon,
            dlist=self.dlist,
            verbose=self.verbose,
        )

        self.optimizer.computeMarginals()

        return self

    def transform(self, X, y, transform_y=False):
        """Transform the input data using the fitted preprocessor.

        Args:
            X: Input features
            y: Target values
            transform_y: Whether to transform the target values as well

        Returns:
            DataFrame: Transformed dataset
        """
        X_df, y_df, sensitive_df = self.createDataframe(X, y)
        X_features = X_df.columns.to_list()
        y_features = y_df.columns.to_list()
        sensitive_names = sensitive_df.columns.to_list()

        combined_df = pd.concat([X_df, sensitive_df], axis=1)
        combined_df = pd.concat([combined_df, y_df], axis=1)

        if transform_y:
            prob_matrix = self.optimizer.dfP.map(lambda x: 0 if x < 1e-8 else x)
            prob_matrix = prob_matrix.divide(prob_matrix.sum(axis=1), axis=0)

            transformed_df = _apply_randomized_mapping(
                combined_df,
                prob_matrix,
                features=sensitive_names + X_features + y_features,
                random_seed=self.seed,
            )
        else:
            # Calculate marginal probabilities excluding target variable
            grouped_probs = (
                self.optimizer.dfFull.reset_index()
                .groupby(sensitive_names + X_features, observed=False)
                .sum()
            )
            marginal_probs = (
                grouped_probs.transpose().reset_index().groupby(X_features, observed=False).sum()
            )
            prob_matrix = marginal_probs.transpose()
            prob_matrix = prob_matrix.drop(y_features, axis=1)
            prob_matrix = prob_matrix.map(lambda x: x if x > 1e-8 else 0)
            prob_matrix = prob_matrix / prob_matrix.sum()

            prob_matrix = prob_matrix.divide(prob_matrix.sum(axis=1), axis=0)

            transformed_df = _apply_randomized_mapping(
                combined_df,
                prob_matrix,
                features=sensitive_names + X_features,
                random_seed=self.seed,
            )
        return transformed_df

    def fit_transform(self, X, y, transform_y=False):
        return self.fit(X, y).transform(X, y, transform_y)


def _apply_randomized_mapping(df, dfMap, features=[], random_seed=None):
    """Apply Randomized mapping to create a new dataframe

    Args:
        df (DataFrame): Input dataframe
        dfMap (DataFrame): Mapping parameters
        features (list): Feature names for which the mapping needs to be applied
        random_seed (int): Random seed

    Returns:
        Perturbed version of df according to the randomizedmapping
    """

    if random_seed is not None:
        np.random.seed(seed=random_seed)

    df2 = df[features].copy()
    rem_cols = [col for col in df.columns if col not in features]
    if rem_cols != []:
        df3 = df[rem_cols].copy()

    idx_list = [tuple(i) for i in df2.itertuples(index=False)]

    draw_probs = dfMap.loc[idx_list]
    draws_possible = draw_probs.columns.tolist()

    # Make random draws - as part of randomizing transformation
    def draw_ind(x):
        return np.random.choice(range(len(draws_possible)), p=x)

    draw_inds = [draw_ind(x) for x in draw_probs.values]

    df2.loc[:, dfMap.columns.names] = [draws_possible[x] for x in draw_inds]

    if rem_cols != []:
        return pd.concat([df2, df3], axis=1)
    else:
        return df2
