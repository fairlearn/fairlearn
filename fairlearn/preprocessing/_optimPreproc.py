import numpy as np
import pandas as pd
from ._optimPreproc_helper import DTools

from sklearn.base import BaseEstimator, TransformerMixin


class OptimizedPreprocessor(BaseEstimator, TransformerMixin):
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
        self.optim_options = {
            "distortion_fun": distortion_function,
            "epsilon": epsilon,
            "clist": clist,
            "dlist": dlist,
        }
        self.verbose = verbose
        self.sensitive_feature_ids = sensitive_feature_ids

    def createDataframe(self, X, y, prefix="column"):
        n_cols = X.shape[1]
        if isinstance(X, pd.DataFrame):
            df_sensitive = X[self.sensitive_feature_ids]
            df_x = X.drop(columns=self.sensitive_feature_ids)
            return df_x, y, df_sensitive
        elif isinstance(X, np.ndarray):
            column_names = [f"{prefix}_{i}" for i in range(1, n_cols + 1)]
            sensitive_columns = []
            for ids in self.sensitive_feature_ids:
                sensitive_columns.append(column_names[ids])
            df_x = pd.DataFrame(X, columns=column_names)
            df_sensitive = df_x[sensitive_columns]
            df_x = df_x.drop(columns=sensitive_columns)
            df_y = pd.DataFrame(y, columns=[f"{prefix}_{n_cols+1}"])
            return df_x, df_y, df_sensitive

    def fit(self, X, y):
        df_x, df_y, df_sensitive = self.createDataframe(X, y)
        X_features = df_x.columns.to_list()
        Y_features = df_y.columns.to_list()
        D_features = df_sensitive.columns.to_list()
        features = X_features + D_features + Y_features
        df = pd.concat([df_x, df_sensitive], axis=1)
        df = pd.concat([df, df_y], axis=1)

        self.opt = DTools(df=df, features=features)
        self.opt.setFeatures(D=D_features, X=X_features, Y=Y_features)
        self.opt.set_distortion(
            self.optim_options["distortion_fun"], self.optim_options["clist"]
        )

        self.opt.optimize(
            epsilon=self.optim_options["epsilon"],
            dlist=self.optim_options["dlist"],
            verbose=self.verbose,
        )

        self.opt.computeMarginals()

        return self

    def transform(self, X, y, transform_y=False):
        df_x, df_y, df_sensitive = self.createDataframe(X, y)
        X_features = df_x.columns.to_list()
        Y_features = df_y.columns.to_list()
        D_features = df_sensitive.columns.to_list()
        df = pd.concat([df_x, df_sensitive], axis=1)
        df = pd.concat([df, df_y], axis=1)

        if transform_y:
            dfP_withY = self.opt.dfP.applymap(lambda x: 0 if x < 1e-8 else x)
            dfP_withY = dfP_withY.divide(dfP_withY.sum(axis=1), axis=0)

            df_transformed = _apply_randomized_mapping(
                df,
                dfP_withY,
                features=D_features + X_features + Y_features,
                random_seed=self.seed,
            )
        else:
            d1 = (
                self.opt.dfFull.reset_index()
                .groupby(D_features + X_features, observed=False)
                .sum()
            )
            d2 = d1.transpose().reset_index().groupby(X_features, observed=False).sum()
            dfP_noY = d2.transpose()
            dfP_noY = dfP_noY.drop(Y_features, axis=1)
            dfP_noY = dfP_noY.applymap(lambda x: x if x > 1e-8 else 0)
            dfP_noY = dfP_noY / dfP_noY.sum()

            dfP_noY = dfP_noY.divide(dfP_noY.sum(axis=1), axis=0)

            df_transformed = _apply_randomized_mapping(
                df,
                dfP_noY,
                features=D_features + X_features,
                random_seed=self.seed,
            )
        return df_transformed

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
