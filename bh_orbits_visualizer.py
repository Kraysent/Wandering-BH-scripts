import glob
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches

from scriptslib import mnras
from scriptslib import plot as splot

INPUT_DIR = "bh_orbit_visualizer/input/{}"
EXTENT = [0, 1, 0, 30]
RESOLUTION_1D = 200


@dataclass
class Params:
    file_pattern: str
    color: str
    name: str


parameters = [
    Params("orbits/i30e35_*", "r", "$35^{\circ}$"),
    Params("orbits/i30e40_*", "b", "$40^{\circ}$"),
    Params("orbits/i30e45_*", "g", "$45^{\circ}$"),
    Params("orbits/i30e50_*", "y", "$50^{\circ}$"),
    Params("orbits/i30e55_*", "m", "$55^{\circ}$"),
]


def show():
    matrices = []
    colors = []

    fig, ax = plt.subplots()
    patches = []

    for param in parameters:
        files = glob.glob(INPUT_DIR.format(param.file_pattern))
        semimajor_axes = np.zeros(len(files))
        eccentricities = np.zeros(len(files))

        for i, file in enumerate(files):
            with open(file, "r") as f:
                sma, ecc = map(lambda x: float(x), f.read().split(","))
                semimajor_axes[i] = sma
                eccentricities[i] = ecc

        sma_filter = semimajor_axes < EXTENT[-1]
        semimajor_axes = semimajor_axes[sma_filter]
        eccentricities = eccentricities[sma_filter]

        print(eccentricities)
        print(semimajor_axes)
        hist = np.zeros((RESOLUTION_1D, RESOLUTION_1D))
        ecc_indices = np.digitize(eccentricities, np.linspace(*EXTENT[:2], RESOLUTION_1D))
        sma_indices = np.digitize(semimajor_axes, np.linspace(*EXTENT[2:], RESOLUTION_1D))

        for i in range(len(semimajor_axes)):
            hist[ecc_indices[i] - 1, sma_indices[i] - 1] += 1

        matrices.append(hist[:, ::-1].T)
        colors.append(param.color)
        patches.append(mpatches.Patch(color=param.color, label=param.name))

    # plt.plot(eccentricities, semimajor_axes, "g.")
    pic_r, pic_g, pic_b = splot.plot_colored_hist(matrices, colors)
    ax.imshow(
        np.stack([pic_r, pic_g, pic_b], axis=2),
        extent=EXTENT,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlabel("Eccentricity", fontsize=mnras.FONT_SIZE)
    ax.set_ylabel("Semi-major axis, kpc", fontsize=mnras.FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)
    ax.legend(handles=patches, loc="lower left", fontsize=mnras.FONT_SIZE)
    # cbar = plt.colorbar(pic)
    # cbar.set_label("Number of models with given orbit", fontsize=mnras.FONT_SIZE)
    # cbar.ax.tick_params(labelsize=mnras.FONT_SIZE)

    plt.show()


if __name__ == "__main__":
    show()
