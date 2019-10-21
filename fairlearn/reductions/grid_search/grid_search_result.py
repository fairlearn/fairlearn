# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GridSearchResult:
    def __init__(self, predictor, lambda_vec, objective, gamma):
        self.predictor = predictor
        self.lambda_vec = lambda_vec
        self.objective = objective
        self.gamma = gamma
