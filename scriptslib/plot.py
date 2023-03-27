import matplotlib.pyplot as plt
import numpy as np


def _log_scale(array: np.ndarray, low: float = 0, high: float = 1, scale_background: bool = False) -> np.ndarray:
    """
    Works only for positive integer arrays!
    Scales array to [low, high] interval with logariphmic scale.
    """
    array[array != 0] = np.log10(array[array != 0])
    array = low + (high - low) / np.max(array) * array

    if not scale_background:
        array[array == low] = 0

    return array


def plot_hist(
    red_x: np.ndarray,
    red_y: np.ndarray,
    green_x: np.ndarray = None,
    green_y: np.ndarray = None,
    blue_x: np.ndarray = None,
    blue_y: np.ndarray = None,
    *,
    extent: tuple[float, float, float, float],
    resolution: int = 500,
    axes=None,
):
    hists = {}

    for c, x, y in zip(["r", "g", "b"], [red_x, green_x, blue_x], [red_y, green_y, blue_y]):
        if (x is None) or (y is None):
            hists[c] = np.zeros((resolution, resolution))
            continue

        hist, _, _ = np.histogram2d(x, y, resolution, [extent[:2], extent[2:]])
        hist = _log_scale(hist.T[::-1, :], low=0.4)
        hists[c] = hist

    rgb_map = np.stack([hists["r"], hists["g"], hists["b"]], axis=2)
    rgb_map[(rgb_map[:, :] ** 2).sum(axis=2) == 0] = 1
    params = dict(extent=extent, interpolation="nearest", aspect="auto")

    return plt.imshow(rgb_map, **params) if axes is None else axes.imshow(rgb_map, **params)
