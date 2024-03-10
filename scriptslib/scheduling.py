import functools
import os
import sys
from concurrent import futures
from contextlib import redirect_stdout
from enum import Enum
from typing import Any, Callable

import numpy as np


class PrintPrepender:
    stdout = sys.stdout

    def __init__(self, text_to_prepend):
        self.text_to_prepend = text_to_prepend
        self.buffer = [self.text_to_prepend]

    def write(self, text):
        lines = text.splitlines(keepends=True)
        for line in lines:
            self.buffer.append(line)
            self.flush()
            if line.endswith(os.linesep):
                self.buffer.append(self.text_to_prepend)

    def flush(self, *args):
        self.stdout.write("".join(self.buffer))
        self.stdout.flush()
        self.buffer.clear()


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


class LinearScheduler:
    """
    Runs `processor` function over a set of values in different modes. 
    If desired, prepends identification to stdout of each `processor` call.
    """
    def __init__(
        self, processor: Callable[..., Any], conditions: np.array, prepend_stdout: bool = True, *args, **kwargs
    ):
        self.processor = processor
        self.conditions = conditions
        self.prepend_stdout = prepend_stdout
        self.args = args
        self.kwargs = kwargs

    def _iterated_processor(self, args):
        ix, x = args

        if not self.prepend_stdout:
            return ix, self.processor(x, *self.args, **self.kwargs)

        buf = PrintPrepender(f"[{ix}: {x}] ")
        with redirect_stdout(buf):
            return ix, self.processor(x, *self.args, **self.kwargs)

    def run(self, mode: SchedulerMode = SchedulerMode.Sequential, workers_num: int = 8) -> np.array:
        params_list = self.conditions
        results = np.zeros(shape=(len(params_list)))

        if mode == SchedulerMode.Sequential:
            for ix, x in enumerate(params_list):
                ix, res = self._iterated_processor((ix, x))
                results[ix] = res
        elif mode == SchedulerMode.Threaded:
            with futures.ThreadPoolExecutor(max_workers=workers_num) as executor:
                for ix, res in executor.map(self._iterated_processor, enumerate(params_list)):
                    results[ix] = res
        elif mode == SchedulerMode.Multiprocessed:
            with futures.ProcessPoolExecutor(max_workers=workers_num) as executor:
                for ix, res in executor.map(self._iterated_processor, enumerate(params_list)):
                    results[ix] = res

        return results
