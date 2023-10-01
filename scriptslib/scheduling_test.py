import pytest
import numpy as np
from scriptslib.scheduling import MapScheduler


def add(x, y):
    return x + y


@pytest.fixture
def scheduler():
    x_conditions = np.array([1, 2, 3])
    y_conditions = np.array([4, 5, 6])
    return MapScheduler(add, x_conditions, y_conditions)


@pytest.mark.parametrize(
    "x_conditions, y_conditions, expected",
    [
        (np.array([1, 2]), np.array([3, 4]), np.array([[4, 5], [5, 6]])),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([[5, 6, 7], [6, 7, 8], [7, 8, 9]])),
    ],
)
def test_run_happy_path(scheduler, x_conditions, y_conditions, expected):
    scheduler.x_conditions = x_conditions
    scheduler.y_conditions = y_conditions

    result = scheduler.run()

    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "x_conditions, y_conditions, expected",
    [
        (np.array([1]), np.array([2]), np.array([[3]])),  # Single condition
    ],
)
def test_run_edge_cases(scheduler, x_conditions, y_conditions, expected):
    scheduler.x_conditions = x_conditions
    scheduler.y_conditions = y_conditions

    result = scheduler.run()

    assert np.array_equal(result, expected)
