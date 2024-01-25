import numpy as np
import pandas as pd
from ._optimPreproc_helper import DTools

from sklearn.base import BaseEstimator, TransformerMixin


class OptimizedPreprocessor:
    def __init__(self, optimizer, optim_options, verbose=False, seed=None) -> None:
        super().__init__()

        self.seed = seed
        self.optimizer = optimizer
        self.optim_options = optim_options
        self.verbose = verbose

    def fit(self, df, X_features, Y_features, D_features):
        self.X_features = X_features
        self.Y_features = Y_features
        self.D_features = D_features
        self.features = self.X_features + self.Y_features + self.D_features

        df = df[self.features]

        self.opt = DTools(df=df, features=self.features)

        self.opt.setFeatures(D=self.D_features, X=self.X_features, Y=self.Y_features)

        self.opt.setDistortion(
            self.optim_options["distortion_fun"], self.optim_options["clist"]
        )

        self.opt.optimize(
            epsilon=self.optim_options["epsilon"],
            dlist=self.optim_options["dlist"],
            verbose=self.verbose,
        )

        self.opt.computeMarginals()

        """
        self.protected_feature_names = protected_features
        self.Y_feature_names = Y_label_name
        self.X_feature_names = [
            n
            for n in df.columns.tolist()
            if n not in self.Y_feature_names and n not in self.protected_feature_names
        ]
        self.feature_names = (
            self.X_feature_names + self.Y_feature_names + self.protected_feature_names
        )
        self.opt = self.optimizer(df=df, features=self.feature_names)
        self.opt.set_features(
            D=self.protected_feature_names,
            X=self.X_feature_names,
            Y=self.Y_feature_names,
        )
        self.opt.set_distortion(
            self.optim_options["distortion_fun"], clist=self.optim_options["clist"]
        )
        self.opt.optimize(
            epsilon=self.optim_options["epsilon"],
            dlist=self.optim_options["dlist"],
            verbose=self.verbose,
        )
        self.opt.compute_marginals()
        """

    def transform(self, df, X_features, Y_features, D_features, transform_Y=False):
        features = X_features + Y_features + D_features

        df = df[features]

        if transform_Y:
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
        """
        d1 = self.opt.dfFull.reset_index().groupby(D_features + X_features).sum()
        d2 = d1.transpose().reset_index().groupby(X_features).sum()
        dTest = d2.transpose()
        dTest = dTest.drop(Y_features, 1)
        dTest = dTest.applymap(lambda x: x if x > 1e-8 else 0)
        dTest = dTest / dTest.sum()

        # this is the dataframe with the randomization for the tranformation set
        dfPtest = dTest.divide(dTest.sum(axis=1), axis=0)

        print("Randomizing test set...")
        df_test_new = randomize(df_test, dfPtest, features=D_features + X_features)

        Y_feature_names = Y_label_name
        D_feature_names = protected_features
        X_feature_names = [
            n
            for n in df.columns.tolist()
            if n not in self.Y_feature_names and n not in D_feature_names
        ]
        if (
            X_feature_names != self.X_feature_names
            or D_feature_names != self.protected_feature_names
        ):
            raise ValueError(
                "The feature names of inputs and protected "
                "attributes must match with the training dataset."
            )

        if transform_Y and (Y_feature_names != self.Y_feature_names):
            raise ValueError(
                "The label name must match with that in the training dataset"
            )

        if transform_Y:
            # randomized mapping when Y is requested to be transformed
            dfP_withY = self.opt.dfP.applymap(lambda x: 0 if x < 1e-8 else x)
            dfP_withY = dfP_withY.divide(dfP_withY.sum(axis=1), axis=0)

            df_transformed = _apply_randomized_mapping(
                df,
                dfP_withY,
                features=D_feature_names + X_feature_names + Y_feature_names,
                random_seed=self.seed,
            )
        else:
            # randomized mapping when Y is not requested to be transformed
            d1 = (
                self.opt.dfFull.reset_index()
                .groupby(D_feature_names + X_feature_names)
                .sum()
            )
            d2 = d1.transpose().reset_index().groupby(X_feature_names).sum()
            dfP_noY = d2.transpose()
            dfP_noY = dfP_noY.drop(Y_feature_names, 1)
            dfP_noY = dfP_noY.applymap(lambda x: x if x > 1e-8 else 0)
            dfP_noY = dfP_noY / dfP_noY.sum()

            dfP_noY = dfP_noY.divide(dfP_noY.sum(axis=1), axis=0)

            df_transformed = _apply_randomized_mapping(
                df,
                dfP_noY,
                features=D_feature_names + X_feature_names,
                random_seed=self.seed,
            )
        return df_transformed
        """


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
