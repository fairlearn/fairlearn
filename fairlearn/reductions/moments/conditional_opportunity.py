# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from .moment import Moment


class ConditionalOpportunity(Moment):
    """Generic fairness metric including DemographicParity and EqualizedOdds"""

    def init(self, dataX, dataA, dataY, dataGrp):
        super().init(dataX, dataA, dataY)
        self.tags["grp"] = dataGrp
        self.prob_grp = self.tags.groupby("grp").size() / self.n
        self.prob_attr_grp = self.tags.groupby(
            ["grp", "protected_attribute"]).size() / self.n
        signed = pd.concat([self.prob_attr_grp, self.prob_attr_grp],
                           keys=["+", "-"],
                           names=["sign", "grp", "protected_attribute"])
        self.index = signed.index

    def gamma(self, predictor):
        """ Calculates the degree to which constraints are currently violated by
        the predictor.
        """
        pred = predictor(self.X)
        self.tags["pred"] = pred
        expect_grp = self.tags.groupby("grp").mean()
        expect_attr_grp = self.tags.groupby(
            ["grp", "protected_attribute"]).mean()
        expect_attr_grp["diff"] = expect_attr_grp["pred"] - expect_grp["pred"]
        g_unsigned = expect_attr_grp["diff"]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=["sign", "grp", "protected_attribute"])
        self._gamma_descr = str(expect_attr_grp[["pred", "diff"]])
        return g_signed

    def lambda_signed(self, lambda_vec):
        return lambda_vec["+"] - lambda_vec["-"]

    def signed_weights(self, lambda_vec):
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        adjust = lambda_signed.sum(level="grp") / self.prob_grp \
            - lambda_signed / self.prob_attr_grp
        signed_weights = self.tags.apply(
            lambda row: adjust[row["grp"], row["protected_attribute"]], axis=1
        )
        return signed_weights


class DemographicParity(ConditionalOpportunity):
    """ Demographic parity
    A classifier h satisfies DemographicParity if
    Prob[h(X) = y' | A = a] = Prob[h(X) = y'] for all a, y'
    """
    short_name = "DemographicParity"

    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataA, dataY,
                     dataY.apply(lambda y: "all"))


class EqualizedOdds(ConditionalOpportunity):
    """ Equalized odds
    Adds conditioning on label compared to Demographic parity, i.e.
    Prob[h(X) = y' | A = a, Y = y] = Prob[h(X) = y' | Y = y] for all a, y, y'
    """
    short_name = "EqualizedOdds"

    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataA, dataY,
                     dataY.apply(lambda y: "label=" + str(y)))


class BoundedGroupLoss(ConditionalOpportunity):
    def init(self, dataX, dataA, dataY):
        raise NotImplementedError("Placeholder for now")
