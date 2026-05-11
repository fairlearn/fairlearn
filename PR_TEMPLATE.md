# Pull Request: Fix fetch_diabetes_hospital as_frame=False Support

## Description
This PR fixes the `fetch_diabetes_hospital()` function to properly support the `as_frame=False` parameter. Previously, this parameter would raise a ValueError due to type mismatch in the OpenML dataset.

## Problem
- Users could not retrieve the diabetes hospital dataset as numpy arrays
- Tests were skipped with a note indicating the dataset needed fixing
- The function was inconsistent with other `fetch_*` functions in the module

## Solution
- Always fetch the dataset as a pandas DataFrame from OpenML (which handles type conversion correctly)
- Convert to numpy arrays when `as_frame=False` is requested
- Properly handle the `return_X_y` parameter for both modes
- Remove the workaround skip conditions from tests

## Type of Change
- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Changes Made
1. **fairlearn/datasets/_fetch_diabetes_hospital.py**
   - Removed warning note about `as_frame=False` from docstring
   - Added internal logic to always fetch as DataFrame
   - Added conversion to numpy arrays when needed
   - Properly handles both `as_frame` and `return_X_y` parameters

2. **test/unit/datasets/test_datasets.py**
   - Removed `pytest.skip()` calls for `fetch_diabetes_hospital` with `as_frame=False`
   - Removed `test_fetch_diabetes_hospital_as_ndarray_raises_value_error()` test
   - Tests now run for all parameter combinations

## Testing
- [x] All existing tests pass
- [x] New functionality tested with both `as_frame=True` and `as_frame=False`
- [x] Tested with `return_X_y=True` and `return_X_y=False`
- [x] No breaking changes to existing API

## Checklist
- [x] My code follows the code style of this project (PEP8, black compatible)
- [x] I have updated the docstring if needed
- [x] I have added/updated tests for my changes
- [x] All tests pass locally
- [x] I have not introduced any breaking changes
- [x] My changes are consistent with other `fetch_*` functions

## Related Issues
Fixes the issue where `fetch_diabetes_hospital(as_frame=False)` would raise ValueError

## Additional Context
This fix enables users to work with the diabetes hospital dataset in numpy format, 
which is useful for workflows that don't use pandas. The implementation follows the 
same pattern as other dataset fetching functions in scikit-learn and fairlearn.

## Commit Message
```
FIX: Support as_frame=False in fetch_diabetes_hospital

- Always fetch dataset as DataFrame from OpenML to handle type conversion
- Convert to numpy arrays when as_frame=False is requested
- Remove pytest.skip() calls that were blocking tests
- Remove test that expected ValueError since issue is now fixed
- Properly handle return_X_y parameter for both frame and array modes
- Maintains API compatibility with other fetch_* functions
```
