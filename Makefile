# simple makefile to simplify repetitive build env management tasks

PYTHON ?= python
PYTEST ?= pytest
SPHINX ?= python -m sphinx

all: clean inplace test-unit

clean:
	git clean -xfd

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-coverage:
	$(PYTEST) test -m "not notebooks" --ignore=test/install --cov=fairlearn --cov-report=xml --cov-report=html

test-unit:
	$(PYTEST) ./test/unit

doc:
	$(SPHINX) -b html -j auto docs docs/_build/html
