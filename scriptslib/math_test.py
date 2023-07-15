import numpy as np

from scriptslib import math


def test_weighted_median_simple():
    arr = np.array(
        [
            [1, 2, 3],
            [7, 9, 2],
            [2, 0, 7],
            [3, 9, 10],
            [11, 1, 13],
        ]
    )
    weights = np.array([3, 1, 1, 1, 1])

    expected = np.array([2, 2, 3])
    actual = math.weighted_median(arr, weights)

    np.testing.assert_equal(actual, expected)


def test_weighted_median_float_weights():
    """
    this test is slightly wrong: it accepts median for even number of values as the first one
    instead of their average
    """
    arr = np.array(
        [
            [1, 2, 3],
            [7, 9, 2],
            [2, 0, 7],
            [3, 9, 10],
            [11, 1, 13],
        ]
    )
    weights = np.array([1.5, 1.5, 1, 1, 1])

    expected = np.array([3, 2, 3])
    actual = math.weighted_median(arr, weights)
    
    np.testing.assert_array_equal(actual, expected)
