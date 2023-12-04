import glob
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches

from scriptslib import mnras
from scriptslib import plot as splot
from scipy import stats

INPUT_DIR = "models_velocity_vector/results/{}"
OUTPUT_DIR = "bh_orbit_visualizer/{}"
EXTENT = [0, 1, 0, 75]
RESOLUTION_1D = 200


@dataclass
class Params:
    file_pattern: str
    color: str
    name: str


parameters = [
    Params("i30e35_*", splot.colors[0], "$35^{\circ}$"),
    Params("i30e40_*", splot.colors[3], "$40^{\circ}$"),
    Params("i30e45_*", splot.colors[1], "$45^{\circ}$"),
    Params("i30e50_*", splot.colors[5], "$50^{\circ}$"),
    Params("i30e55_*", splot.colors[2], "$55^{\circ}$"),
]


def show():
    matrices = []
    colors = []

    fig, ax = plt.subplots()
    legend_patches = []

    for param in parameters:
        files = glob.glob(INPUT_DIR.format(param.file_pattern))
        semimajor_axes = np.zeros(len(files))
        eccentricities = np.zeros(len(files))

        for i, file in enumerate(files):
            with open(file, "r") as f:
                sma, ecc = map(lambda x: float(x), f.read().split(","))
                semimajor_axes[i] = sma
                eccentricities[i] = ecc

        ecc_mean, ecc_std = np.median(eccentricities), stats.median_abs_deviation(eccentricities)
        sma_mean, sma_std = np.median(semimajor_axes), stats.median_abs_deviation(semimajor_axes)
        print(f"{param}")
        print(f"{ecc_mean} +- {ecc_std}")
        print(f"{sma_mean} +- {sma_std}")

        sma_filter = semimajor_axes < EXTENT[-1]
        semimajor_axes = semimajor_axes[sma_filter]
        eccentricities = eccentricities[sma_filter]

        hist = np.zeros((RESOLUTION_1D * 2, RESOLUTION_1D))
        ecc_indices = np.digitize(eccentricities, np.linspace(*EXTENT[:2], RESOLUTION_1D * 2))
        sma_indices = np.digitize(semimajor_axes, np.linspace(*EXTENT[2:], RESOLUTION_1D))

        for i in range(len(semimajor_axes)):
            hist[ecc_indices[i] - 1, sma_indices[i] - 1] += 1

        # make the picture less noisy. If the pixel contains only one result, ignore it.
        hist[hist <= 1] = 0

        matrices.append(hist[:, ::-1].T)
        colors.append(param.color)
        legend_patches.append(mpatches.Patch(color=param.color, label=param.name))

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
    ax.legend(handles=legend_patches, loc="lower left", fontsize=mnras.FONT_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 75)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR.format("predictions.pdf"))


if __name__ == "__main__":
    show()
