import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib import colors as mpl_colors
from matplotlib.image import AxesImage
from scipy.ndimage import filters

colors = seaborn.color_palette("viridis")


def _log_scale(array: np.ndarray, low: float = 0, high: float = 1, scale_background: bool = False) -> np.ndarray:
    """
    Works only for positive integer arrays!
    Scales array to [low, high] interval with logarithmic scale.
    """
    if array.max() == 0:
        return array

    array[array != 0] = np.log10(array[array != 0])

    if array.max() != 0:
        array = low + (high - low) / array.max() * array

    # if not scale_background:
    #     array[array == low] = 0

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
    red_scale: float = 1,
    green_scale: float = 1,
    blue_scale: float = 1,
    gauss_filter_sigma: float | None = None,
    return_rgbmap: bool = False,
) -> AxesImage | tuple[AxesImage, np.ndarray]:
    hists = {}
    low = 0.7

    for c, x, y in zip(["r", "g", "b"], [red_x, green_x, blue_x], [red_y, green_y, blue_y]):
        if (x is None) or (y is None):
            hists[c] = np.zeros((resolution, resolution))
            continue

        heatmap, _, _ = np.histogram2d(x, y, resolution, [extent[:2], extent[2:]])
        heatmap = _log_scale(heatmap.T[::-1, :], low=low)
        if gauss_filter_sigma is not None:
            heatmap = filters.gaussian_filter(heatmap, sigma=gauss_filter_sigma)
        heatmap[np.abs(heatmap - low) < 0.000001] = 0
        hists[c] = heatmap

    rgb_map = np.stack(
        [
            hists["r"] * red_scale,
            hists["g"] * green_scale,
            hists["b"] * blue_scale,
        ],
        axis=2,
    )
    rgb_map[(rgb_map[:, :] ** 2).sum(axis=2) == 0] = 1
    params = dict(extent=extent, interpolation="nearest", aspect="auto")
    image = plt.imshow(rgb_map, **params) if axes is None else axes.imshow(rgb_map, **params)

    return (image, rgb_map) if return_rgbmap else image


def plot_colored_hist(maps: list[np.ndarray], colors: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(maps) == len(colors)

    colors = [mpl_colors.to_rgb(c) for c in colors]
    total_map = sum(maps)

    r_map = sum(maps[i] * colors[i][0] * (1 - maps[i] / maps[i].max()) for i in range(len(maps))) / total_map
    g_map = sum(maps[i] * colors[i][1] * (1 - maps[i] / maps[i].max()) for i in range(len(maps))) / total_map
    b_map = sum(maps[i] * colors[i][2] * (1 - maps[i] / maps[i].max()) for i in range(len(maps))) / total_map

    r_map[total_map == 0] = 1
    g_map[total_map == 0] = 1
    b_map[total_map == 0] = 1

    return r_map, g_map, b_map


def plot_colored_points(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    colors: list[str],
    *,
    extent: tuple[float, float, float, float],
    resolution: int = 500,
    gauss_filter_sigma: float | None = None,
    low=0.7,
):
    assert len(xs) == len(ys)
    hists = []

    for x, y, c in zip(xs, ys, colors):
        heatmap, _, _ = np.histogram2d(x, y, resolution, [extent[:2], extent[2:]])
        # heatmap = _log_scale(heatmap.T[::-1, :], low=low)
        if gauss_filter_sigma is not None:
            heatmap = filters.gaussian_filter(heatmap, sigma=gauss_filter_sigma)
        heatmap[np.abs(heatmap - low) < 0.000001] = 0
        hists.append(heatmap)

    return plot_colored_hist(hists, colors)
