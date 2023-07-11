from typing import List

from amuse.lab import VectorQuantity
import numpy as np


def get_lengths(value_vectors: VectorQuantity | np.ndarray, axis: int = 1) -> VectorQuantity:
    return (value_vectors**2).sum(axis=axis) ** 0.5


def sort_with(values1: VectorQuantity, *values: list[VectorQuantity]) -> list[VectorQuantity]:
    """
    Sorts first array and then applies thew same permutation for all the other arrays
    """
    perm = values1.argsort()
    sorted_values1 = values1[perm]
    sorted_values = [value[perm] for value in values]

    return [sorted_values1, *sorted_values]
