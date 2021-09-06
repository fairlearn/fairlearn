# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
import math
from typing import Generator, Iterable, List

import fairlearn.metrics as metrics


def _get_raw_MetricFrame():
    # Gets an uninitialised MetricFrame for testing purposes
    return metrics.MetricFrame.__new__(metrics.MetricFrame)


def batchify(num_batches, *args: List[Iterable]) -> Generator:
    reference = args[0]
    assert [len(arr) == len(reference) for arr in args], "Can't batch arrays of unequal lengths."
    assert num_batches <= len(reference), "Can't make more batches than there is elements."
    batch_size = int(math.ceil(len(reference) / num_batches))
    for idx in range(0, len(reference), batch_size):
        yield [arr[idx: (idx + batch_size)] for arr in args]
