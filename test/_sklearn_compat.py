# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Compatibility shim around :func:`sklearn.utils.estimator_checks.parametrize_with_checks`.

sklearn's helper internally calls ``pytest.mark.parametrize("estimator, check",
_checks_generator(...))`` with a generator. pytest 9 emits
``PytestRemovedIn10Warning`` for non-Collection iterables passed to ``parametrize``,
and pytest 10 will reject them outright. Combined with
``filterwarnings = ["error"]`` in :file:`pyproject.toml` this turns into a
collection failure for every test file that uses sklearn's helper.

The wrapper below preserves sklearn's signature so call sites just need to swap
the import. It consumes the generator upfront and re-emits the same
``pytest.mark.parametrize`` with a materialized list.
"""

import pytest
from sklearn.utils.estimator_checks import (
    parametrize_with_checks as _sk_parametrize_with_checks,
)


def parametrize_with_checks(estimators, **kwargs):
    """Wrap sklearn's ``parametrize_with_checks`` and materialize its generator.

    Parameters
    ----------
    estimators : list of estimator instances
        Forwarded to :func:`sklearn.utils.estimator_checks.parametrize_with_checks`.
    **kwargs
        Forwarded unchanged (e.g. ``legacy``, ``expected_failed_checks``,
        ``xfail_strict``).

    Returns
    -------
    pytest.MarkDecorator
        A ``pytest.mark.parametrize`` decorator whose argvalues are a list
        instead of a generator.
    """
    decorator = _sk_parametrize_with_checks(estimators, **kwargs)
    mark = decorator.mark
    argnames, argvalues = mark.args
    return pytest.mark.parametrize(argnames, list(argvalues), **mark.kwargs)
