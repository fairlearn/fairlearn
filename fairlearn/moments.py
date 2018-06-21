# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

__all__ = ["DP", "EO"]

class Moment:
    """Generic moment"""

    def __init__(self):
        self.initialized = False

    def init(self, dataX, dataA, dataY):
        assert self.initialized==False, \
            "moments can be initialized only once"
        self.X = dataX
        self.tags = pd.DataFrame({"attr": dataA, "label": dataY})
        self.n = dataX.shape[0]
        self.initialized = True
        self._gamma_descr = None


class MisclassError(Moment):
    """Misclassification error"""
    short_name = "Err"

    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataA, dataY)
        self.index = ["all"]

    def gamma(self, predictor):
        pred = predictor(self.X)
        error = pd.Series(data=(self.tags["label"]-pred).abs().mean(),
                          index=self.index)
        self._gamma_descr = str(error)
        return error

    def signed_weights(self, lambda_vec=None):
        if lambda_vec is None:
            return 2*self.tags["label"]-1
        else:
            return lambda_vec["all"]*(2*self.tags["label"]-1)


class _CondOpportunity(Moment):
    """Generic fairness metric including DP and EO"""

    def init(self, dataX, dataA, dataY, dataGrp):
        super().init(dataX, dataA, dataY)
        self.tags["grp"] = dataGrp
        self.prob_grp = self.tags.groupby("grp").size()/self.n
        self.prob_attr_grp = self.tags.groupby(["grp", "attr"]).size()/self.n
        signed = pd.concat([self.prob_attr_grp, self.prob_attr_grp],
                           keys=["+", "-"],
                           names=["sign", "grp", "attr"])
        self.index = signed.index
        
    def gamma(self, predictor):
        pred = predictor(self.X)
        self.tags["pred"] = pred
        expect_grp = self.tags.groupby("grp").mean()
        expect_attr_grp = self.tags.groupby(["grp", "attr"]).mean()
        expect_attr_grp["diff"] = expect_attr_grp["pred"] - expect_grp["pred"]
        g_unsigned = expect_attr_grp["diff"]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+","-"],
                             names=["sign", "grp", "attr"])
        self._gamma_descr = str(expect_attr_grp[["pred", "diff"]])
        return g_signed

    def signed_weights(self, lambda_vec):
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        adjust = lambda_signed.sum(level="grp")/self.prob_grp \
                 - lambda_signed/self.prob_attr_grp
        signed_weights = self.tags.apply(
            lambda row: adjust[row["grp"], row["attr"]], axis=1
        )
        return signed_weights
    
    
class DP(_CondOpportunity):
    """Demographic parity"""
    short_name = "DP"

    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataA, dataY,
                     dataY.apply(lambda y : "all"))

class EO(_CondOpportunity):
    """Equalized odds"""
    short_name = "EO"
    
    def init(self, dataX, dataA, dataY):
        super().init(dataX, dataA, dataY,
                     dataY.apply(lambda y : "label="+str(y)))
