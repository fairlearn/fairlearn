# simple makefile to simplify repetitive build env management tasks

test-coverage:
	python -m pytest test -m "not notebooks" --ignore=test/perf --ignore=test/install --cov=fairlearn --cov-report=xml --cov-report=html

test-unit:
	python -m pytest ./test/unit

test-perf:
	python -m pytest ./test/perf