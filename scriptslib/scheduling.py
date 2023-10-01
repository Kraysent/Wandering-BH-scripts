from concurrent import futures
from enum import Enum
import functools
from typing import Any, Callable

import numpy as np


class SchedulerMode(Enum):
    Sequential = 1
    Threaded = 2
    Multiprocessed = 3


class MapScheduler:
    """
    MapScheduler is responsible for scheduling and running a `processor` function on a grid of x and y conditions.
    """

    def __init__(self, processor: Callable[..., Any], x_conditions: np.array, y_conditions: np.array, *args, **kwargs):
        self.processor = processor
        self.x_conditions = x_conditions
        self.y_conditions = y_conditions
        self.args = args
        self.kwargs = kwargs

    def _iterated_processor(self, args):
        iy, ix, x, y = args
        return iy, ix, self.processor(x, y, *self.args, **self.kwargs)

    def run(self, mode: SchedulerMode = SchedulerMode.Sequential, workers_num: int = 8) -> np.array:
        xv, yv = np.meshgrid(self.x_conditions, self.y_conditions, indexing="ij")
        results = np.zeros(shape=xv.shape)

        params_list = [(iy, ix, xv[iy, ix], yv[iy, ix]) for iy, ix in np.ndindex(xv.shape)]

        if mode == SchedulerMode.Sequential:
            for iy, ix, x, y in params_list:
                iy, ix, res = self._iterated_processor((ix, iy, x, y))
                results[iy, ix] = res
        elif mode == SchedulerMode.Threaded:
            with futures.ThreadPoolExecutor(max_workers=workers_num) as executor:
                for iy, ix, res in executor.map(self._iterated_processor, params_list):
                    results[iy, ix] = res
        elif mode == SchedulerMode.Multiprocessed:
            with futures.ProcessPoolExecutor(max_workers=workers_num) as executor:
                for iy, ix, res in executor.map(self._iterated_processor, params_list):
                    results[iy, ix] = res

        return results
