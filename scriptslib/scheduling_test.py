import numpy as np
import pytest

from scriptslib.scheduling import LinearScheduler, MapScheduler


def add(x, y):
    return x + y

def double(x):
    return x * 2

def get_map_scheduler() -> MapScheduler:
    return MapScheduler(add, np.array([1, 2, 3]), np.array([4, 5, 6]))

@pytest.mark.parametrize(
    "scheduler, x_conditions, y_conditions, expected",
    [
        (get_map_scheduler(), np.array([1, 2]), np.array([3, 4]), np.array([[4, 5], [5, 6]])),
        (get_map_scheduler(), np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([[5, 6, 7], [6, 7, 8], [7, 8, 9]])),
    ],
)
def test_map_run_happy_path(scheduler, x_conditions, y_conditions, expected):
    scheduler.x_conditions = x_conditions
    scheduler.y_conditions = y_conditions

    result = scheduler.run()

    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "scheduler, x_conditions, y_conditions, expected",
    [
        (get_map_scheduler(), np.array([1]), np.array([2]), np.array([[3]])),  # Single condition
    ],
)
def test_map_run_edge_cases(scheduler, x_conditions, y_conditions, expected):
    scheduler.x_conditions = x_conditions
    scheduler.y_conditions = y_conditions

    result = scheduler.run()

    assert np.array_equal(result, expected)

def get_linear_scheduler():
    return LinearScheduler(double, np.array([]))

@pytest.mark.parametrize(
    "scheduler, conditions, expected",
    [
        (get_linear_scheduler(), np.array([1, 2]), np.array([2, 4])),
        (get_linear_scheduler(), np.array([1, 2, 3, -5]), np.array([2, 4, 6, -10])),
    ],
)
def test_map_run_happy_path(scheduler, conditions, expected):
    scheduler.conditions = conditions

    result = scheduler.run()

    assert np.array_equal(result, expected)
