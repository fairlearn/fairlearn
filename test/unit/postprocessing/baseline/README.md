To regenerate the baseline plots run the following command from the root directory of the repository

```
python -m pytest .\test\unit\postprocessing\test_plots.py --mpl-generate-path=test/unit/postprocessing/baseline
```

Make sure to have `pytest-mpl` installed or this will not work.
