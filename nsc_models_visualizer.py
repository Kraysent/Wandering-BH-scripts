from math import floor, log10

import numpy as np
from matplotlib import pyplot as plt

from scriptslib import mnras

RESULTS_DIR = "nsc_models/results/{}"


def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0


def fman(f):
    return f / 10 ** fexp(f)


def contour_level_fmt(x):
    return f"${fman(x):.02f} \cdot 10^{{{fexp(x)}}}$ MSun"


if __name__ == "__main__":
    data = np.genfromtxt(RESULTS_DIR.format("results.csv"))

    masses = np.linspace(1e5, 5e7, 30)
    sizes = np.linspace(10.0, 50.0, 30)
    levels = np.geomspace(min(masses), max(masses)/2, 10)

    xv, yv = np.meshgrid(masses, sizes, indexing="ij")

    fig, ax = plt.subplots()
    fig.set_size_inches(mnras.size_from_aspect(1))
    fig.subplots_adjust(wspace=0, hspace=0)

    contour = ax.contour(xv, yv, data, levels=levels)
    ax.clabel(
        contour,
        inline=True,
        fmt=contour_level_fmt,
        fontsize=mnras.FONT_SIZE * 0.75,
        inline_spacing=5,
    )

    ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)
    ax.minorticks_on()
    ax.set_xlabel("Initial mass, MSun", fontsize=mnras.FONT_SIZE)
    ax.set_ylabel("Initial scale size, pc", fontsize=mnras.FONT_SIZE)
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR.format("nsc_mass_contours.pdf"), pad_inches=0, bbox_inches="tight")
