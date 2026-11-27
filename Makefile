# simple makefile to simplify repetitive build env management tasks

UV ?= uv
PYTHON ?= $(UV) run --locked python
PYTEST ?= $(PYTHON) -m pytest
SPHINX ?= $(UV) run --locked --no-default-groups --group docs python -m sphinx

all: clean inplace test-unit

clean:
	git clean -xfd

in: inplace # just a shortcut
inplace:
	$(UV) sync --locked

test-coverage:
	$(PYTEST) test -m "not notebooks" --ignore=test/install --cov=fairlearn --cov-report=xml --cov-report=html

test-unit:
	$(PYTEST) ./test/unit

test-deprecation:
	$(PYTEST) -Werror::DeprecationWarning -Werror::FutureWarning ./test/unit

doc:
	$(SPHINX) -v -b html -n -j auto docs docs/_build/html
