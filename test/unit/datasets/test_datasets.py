# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import Bunch

from fairlearn.datasets import (
    fetch_acs_income,
    fetch_adult,
    fetch_bank_marketing,
    fetch_boston,
    fetch_credit_card,
    fetch_diabetes_hospital,
)
from test.utils import DATA_HOME

# =============================================


class TestFairlearnDataset:
    @pytest.mark.parametrize(
        ("module", "loader", "data_id", "names"),
        [
            (
                "fairlearn.datasets._fetch_bank_marketing",
                fetch_bank_marketing,
                1461,
                [
                    "age",
                    "job",
                    "marital",
                    "education",
                    "default",
                    "balance",
                    "housing",
                    "loan",
                    "contact",
                    "day",
                    "month",
                    "duration",
                    "campaign",
                    "pdays",
                    "previous",
                    "poutcome",
                ],
            ),
            (
                "fairlearn.datasets._fetch_credit_card",
                fetch_credit_card,
                42477,
                [
                    "LIMIT_BAL",
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "AGE",
                    "PAY_0",
                    "PAY_2",
                    "PAY_3",
                    "PAY_4",
                    "PAY_5",
                    "PAY_6",
                ]
                + [f"BILL_AMT{i}" for i in range(1, 7)]
                + [f"PAY_AMT{i}" for i in range(1, 7)],
            ),
        ],
    )
    def test_dataset_authoritative_feature_names_without_network(
        self, module, loader, data_id, names
    ):
        data = pd.DataFrame(
            np.zeros((2, len(names))), columns=[f"old{i}" for i in range(len(names))]
        )
        target = pd.Series(["no", "yes"], name="target")
        bunch = Bunch(
            data=data,
            target=target,
            frame=pd.concat([data, target], axis=1),
            feature_names=list(data.columns),
        )
        with patch(f"{module}.fetch_openml", return_value=bunch) as mocked:
            result = loader(as_frame=True)
        assert result.feature_names == names
        assert result.data.columns.tolist() == names
        assert result.frame.columns.tolist() == names + ["target"]
        assert result.target.tolist() == ["no", "yes"]
        assert mocked.call_args.kwargs["data_id"] == data_id

    @pytest.mark.parametrize(
        ("module", "loader", "data_id", "names"),
        [
            (
                "fairlearn.datasets._fetch_bank_marketing",
                fetch_bank_marketing,
                1461,
                [
                    "age",
                    "job",
                    "marital",
                    "education",
                    "default",
                    "balance",
                    "housing",
                    "loan",
                    "contact",
                    "day",
                    "month",
                    "duration",
                    "campaign",
                    "pdays",
                    "previous",
                    "poutcome",
                ],
            ),
            (
                "fairlearn.datasets._fetch_credit_card",
                fetch_credit_card,
                42477,
                [
                    "LIMIT_BAL",
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "AGE",
                    "PAY_0",
                    "PAY_2",
                    "PAY_3",
                    "PAY_4",
                    "PAY_5",
                    "PAY_6",
                ]
                + [f"BILL_AMT{i}" for i in range(1, 7)]
                + [f"PAY_AMT{i}" for i in range(1, 7)],
            ),
        ],
    )
    def test_dataset_feature_metadata_keeps_ndarray_and_return_xy_contract(
        self, module, loader, data_id, names
    ):
        array = np.zeros((2, len(names)))
        target = np.array([0, 1])
        bunch = Bunch(
            data=array, target=target, feature_names=[f"old{i}" for i in range(len(names))]
        )
        with patch(f"{module}.fetch_openml", return_value=bunch):
            result = loader(as_frame=False)
        assert result.feature_names == names
        assert isinstance(result.data, np.ndarray)
        assert result.target.tolist() == [0, 1]
        frame = pd.DataFrame(array, columns=[f"old{i}" for i in range(len(names))])
        with patch(f"{module}.fetch_openml", return_value=(frame, target)):
            X, y = loader(as_frame=True, return_X_y=True)
        assert X.columns.tolist() == names
        assert y.tolist() == [0, 1]
        with patch(f"{module}.fetch_openml", return_value=(array, target)):
            X, y = loader(as_frame=False, return_X_y=True)
        assert isinstance(X, np.ndarray)
        assert y.tolist() == [0, 1]

    @pytest.mark.parametrize(
        ("module", "loader", "names"),
        [
            (
                "fairlearn.datasets._fetch_bank_marketing",
                fetch_bank_marketing,
                [
                    "age",
                    "job",
                    "marital",
                    "education",
                    "default",
                    "balance",
                    "housing",
                    "loan",
                    "contact",
                    "day",
                    "month",
                    "duration",
                    "campaign",
                    "pdays",
                    "previous",
                    "poutcome",
                ],
            ),
            (
                "fairlearn.datasets._fetch_credit_card",
                fetch_credit_card,
                [
                    "LIMIT_BAL",
                    "SEX",
                    "EDUCATION",
                    "MARRIAGE",
                    "AGE",
                    "PAY_0",
                    "PAY_2",
                    "PAY_3",
                    "PAY_4",
                    "PAY_5",
                    "PAY_6",
                ]
                + [f"BILL_AMT{i}" for i in range(1, 7)]
                + [f"PAY_AMT{i}" for i in range(1, 7)],
            ),
        ],
    )
    def test_non_frame_categories_follow_renamed_feature_metadata(self, module, loader, names):
        original_names = [f"V{i}" for i in range(len(names))]
        categories = {"V0": ["low", "high"], "unrelated": ["keep"]}
        bunch = Bunch(
            data=np.zeros((2, len(names))),
            target=np.array([0, 1]),
            feature_names=original_names,
            categories=categories,
        )

        with patch(f"{module}.fetch_openml", return_value=bunch):
            result = loader(as_frame=False)

        assert result.categories == {names[0]: ["low", "high"], "unrelated": ["keep"]}
        assert result.feature_names == names

    @pytest.mark.parametrize(
        ("module", "loader", "data_id"),
        [
            ("fairlearn.datasets._fetch_bank_marketing", fetch_bank_marketing, 1461),
            ("fairlearn.datasets._fetch_credit_card", fetch_credit_card, 42477),
        ],
    )
    @pytest.mark.parametrize(
        ("as_frame", "return_X_y"), [(True, False), (False, False), (True, True), (False, True)]
    )
    def test_dataset_loader_refuses_to_rename_a_record_of_unexpected_width(
        self, module, loader, data_id, as_frame, return_X_y
    ):
        n_columns = 3
        columns = [f"old{i}" for i in range(n_columns)]
        frame = pd.DataFrame(np.zeros((2, n_columns)), columns=columns)
        target = pd.Series([0, 1], name="target")
        if return_X_y:
            payload = (frame, target) if as_frame else (frame.to_numpy(), target.to_numpy())
        else:
            payload = Bunch(
                data=frame if as_frame else frame.to_numpy(),
                target=target,
                frame=pd.concat([frame, target], axis=1),
                feature_names=columns,
            )
        expected = rf"data_id={data_id} .* to have \d+ features, but it has {n_columns}\."
        with (
            patch(f"{module}.fetch_openml", return_value=payload),
            pytest.raises(ValueError, match=expected),
        ):
            loader(as_frame=as_frame, return_X_y=return_X_y)

    @pytest.mark.openml
    @pytest.mark.parametrize("as_frame", [True, False])
    @pytest.mark.parametrize(
        "fetch_function",
        [
            fetch_acs_income,
            fetch_adult,
            fetch_bank_marketing,
            fetch_boston,
            fetch_credit_card,
            fetch_diabetes_hospital,
        ],
    )
    def test_dataset_as_bunch(self, as_frame, fetch_function):
        dataset = fetch_function(as_frame=as_frame, data_home=DATA_HOME)
        assert dataset is not None
        assert dataset["data"].shape is not None
        assert isinstance(dataset["data"], pd.DataFrame if as_frame else np.ndarray)
        assert dataset["target"].shape is not None
        assert isinstance(dataset["target"], pd.Series if as_frame else np.ndarray)
        assert dataset["feature_names"] is not None
        assert isinstance(dataset["feature_names"], list)
        assert dataset["DESCR"] is not None
        assert isinstance(dataset["DESCR"], str)

    @pytest.mark.openml
    @pytest.mark.parametrize("as_frame", [True, False])
    @pytest.mark.parametrize(
        "fetch_function",
        [
            fetch_acs_income,
            fetch_adult,
            fetch_bank_marketing,
            fetch_boston,
            fetch_credit_card,
            fetch_diabetes_hospital,
        ],
    )
    def test_dataset_as_X_y(self, as_frame, fetch_function):
        X, y = fetch_function(as_frame=as_frame, return_X_y=True, data_home=DATA_HOME)
        assert X is not None
        assert isinstance(X, pd.DataFrame if as_frame else np.ndarray)
        assert y is not None
        assert isinstance(y, pd.Series if as_frame else np.ndarray)

    @pytest.mark.openml
    def test_fetch_acs_income_dataframe(self):
        dataset = fetch_acs_income(data_home=DATA_HOME)
        expected_columns = [
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "OCCP",
            "POBP",
            "RELP",
            "WKHP",
            "SEX",
            "RAC1P",
        ]
        assert dataset["data"].columns.to_list() == expected_columns
        assert dataset["data"].shape == (1664500, 10)

    def test_fetch_acs_income_value_error(self):
        with pytest.raises(ValueError):
            fetch_acs_income(states=["XY"], data_home=DATA_HOME)
