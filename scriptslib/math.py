import numpy as np
from amuse.lab import VectorQuantity


def get_lengths(value_vectors: VectorQuantity | np.ndarray, axis: int = 1) -> VectorQuantity:
    return (value_vectors**2).sum(axis=axis) ** 0.5


def sort_with(values1: VectorQuantity, *values: list[VectorQuantity]) -> list[VectorQuantity]:
    """
    Sorts first array and then applies the same permutation for all the other arrays
    """
    perm = values1.argsort()
    sorted_values1 = values1[perm]
    sorted_values = [value[perm] for value in values]

    return [sorted_values1, *sorted_values]


def weighted_median_1d(values, weights) -> float:
    sorted_indices = values.argsort()
    cum_sum = weights[sorted_indices].cumsum()
    cutoff = 0.5 * cum_sum[-1]

    return values[sorted_indices][cum_sum >= cutoff][0]


def weighted_median(values, weights, axis=0):
    return np.apply_along_axis(lambda a: weighted_median_1d(a, weights), arr=values, axis=axis)
