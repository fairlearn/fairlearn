# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


"""This module contains datasets that can be used for benchmarking and education."""


from ._fetch_acs_income import fetch_acs_income
from ._fetch_adult import fetch_adult
from ._fetch_bank_marketing import fetch_bank_marketing
from ._fetch_boston import fetch_boston
from ._fetch_credit_card import fetch_credit_card
from ._fetch_diabetes_hospital import fetch_diabetes_hospital

__all__ = [
    "fetch_acs_income",
    "fetch_adult",
    "fetch_bank_marketing",
    "fetch_boston",
    "fetch_credit_card",
    "fetch_diabetes_hospital",
]
