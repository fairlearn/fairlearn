# Generating baseline plots

To regenerate the baseline plots run the following command from the root directory of the repository

```
python -m pytest .\test\unit\postprocessing\test_plots.py --mpl-generate-path=test/unit/postprocessing/baseline
```

Make sure to have `pytest-mpl` installed or this will not work.

# Running tests with pytest-mpl

`pytest` can run the tests either to check that there are no exceptions (using a typical `pytest` command without extra options) or to actually compare the generated images with the baseline plots (using `pytest --mpl`).
