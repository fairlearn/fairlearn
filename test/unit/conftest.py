# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

# The following pytest configurations are meant to allow silent skipping of tests for scenarios
# that are not meant to happen. We don't want them to show up as skipped.


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "uncollect_if(*, func): function to unselect tests from parametrization",
    )


def pytest_collection_modifyitems(config, items):
    removed = []
    kept = []
    for item in items:
        marker = item.get_closest_marker("uncollect_if")
        if marker:
            func = marker.kwargs["func"]
            if func(**item.callspec.params):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
