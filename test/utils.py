from os import environ


def is_ci() -> bool:
    # https://docs.github.com/en/actions/reference/workflows-and-actions/variables#default-environment-variables
    return environ.get("CI", "") == "true" and environ.get("GITHUB_ACTIONS", "") == "true"


DATA_HOME = environ.get("DATA_HOME", None) if is_ci() else None
