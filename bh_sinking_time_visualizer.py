import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

from scriptslib import mnras
from scriptslib import plot as splot

SMA_MAX = 30  # kpc
MAX_TIME = 13.7  # Gyr
PATH_PREFIX = "rotated/"

RESULTS_DIR = "bh_sinking_times/results/{}"
MODELS_DIR = "bh_sinking_times/models/{}"


def contour_level_fmt(x):
    return f"{x} Gyr"


def contour_level_fmt_smaller(x):
    return f"{x}\nGyr"


@dataclass
class ParameterSet:
    mass: float
    color: tuple[int, int, int]
    label: str
    linestyle: str
    levels: list[float]
    path_prefix: str
    level_formatter: Callable


parameters = [
    ParameterSet(
        2e6,
        splot.colors[0],
        "BH on inclined orbit, M = $2 \cdot 10^6\ M_{\odot}$",
        "solid",
        [13.0],
        "rotated/",
        contour_level_fmt_smaller,
    ),
    ParameterSet(
        1e7,
        splot.colors[5],
        "BH on inclined orbit, M = $10^7\ M_{\odot}$",
        "solid",
        [8.0, 13.0],
        "rotated/",
        contour_level_fmt,
    ),
    ParameterSet(
        1e8,
        splot.colors[2],
        "BH on inclined orbit, M = $10^8\ M_{\odot}$",
        "solid",
        [3.0, 8.0, 13.0],
        "rotated/",
        contour_level_fmt,
    ),
    ParameterSet(
        2e6,
        splot.colors[0],
        "BH in disk, M = $2 \cdot 10^6\ M_{\odot}$",
        "dotted",
        [13.0],
        "in_plane/",
        contour_level_fmt_smaller,
    ),
    ParameterSet(
        1e7,
        splot.colors[5],
        "BH in disk, M = $10^7\ M_{\odot}$",
        "dotted",
        [8.0, 13.0],
        "in_plane/",
        contour_level_fmt,
    ),
    ParameterSet(
        1e8,
        splot.colors[2],
        "BH in disk, M = $10^8\ M_{\odot}$",
        "dotted",
        [3.0, 8.0, 13.0],
        "in_plane/",
        contour_level_fmt,
    ),
]


def prepare_axes(ax):
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, SMA_MAX)
    ax.set_xlabel("Eccentricity", fontsize=mnras.FONT_SIZE)
    ax.set_ylabel("Semi-major axis, kpc", fontsize=mnras.FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)


def display(additional_results: str | None = None):
    fig, ax = plt.subplots()
    fig.set_size_inches(mnras.size_from_aspect(1))
    prepare_axes(ax)
    legend_patches = []

    for params in parameters:
        sinking_times = np.genfromtxt(
            RESULTS_DIR.format(f"{params.path_prefix}bound_time_{params.mass:.2E}_20.csv"), delimiter=","
        )
        ecc_span = np.linspace(0, 1, sinking_times.shape[0])
        sma_span = np.linspace(0, SMA_MAX, sinking_times.shape[1])
        eccs, smas = np.meshgrid(ecc_span, sma_span, indexing="ij")

        contour = ax.contour(
            eccs,
            smas,
            sinking_times,
            colors=[params.color],
            linestyles=params.linestyle,
            levels=params.levels,
        )
        ax.clabel(
            contour,
            inline=True,
            fmt=params.level_formatter,
            fontsize=mnras.FONT_SIZE * 0.75,
            inline_spacing=0,
        )
        legend_patches.append(Line2D([0], [0], color=params.color, label=params.label, linestyle=params.linestyle))

        if additional_results is not None:
            with open(MODELS_DIR.format(additional_results), "r") as j:
                results = json.loads(j.read())

            ecc, sma, colors, markers, fills = (
                results["eccentricities"],
                results["majsemiaxes"],
                results["colors"],
                results["markers"],
                results["fills"],
            )

            for i in range(len(ecc)):
                ax.scatter(
                    ecc[i],
                    sma[i],
                    c=colors[i],
                    edgecolor="black",
                    marker=MarkerStyle(markers[i], fillstyle=fills[i]),
                    s=80,
                )

    ax.legend(
        handles=legend_patches,
        prop={"size": mnras.FONT_SIZE},
        loc="upper left",
    )
    plt.tight_layout()

    plt.savefig(RESULTS_DIR.format("bh_sinking_times.pdf"))


if __name__ == "__main__":
    display()
