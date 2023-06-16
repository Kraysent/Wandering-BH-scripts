import matplotlib.pyplot as plt
import scriptslib
from scriptslib import plot as splot
from amuse.lab import units
import glob
import numpy as np

from scriptslib import mnras

INPUT_DIR = "bh_orbit_visualizer/input/{}"
EXTENT = [0, 1, 0, 25]
RESOLUTION_1D = 150


def show():
    files = glob.glob(INPUT_DIR.format("orbits/*"))
    semimajor_axes = np.zeros(len(files))
    eccentricities = np.zeros(len(files))

    for i, file in enumerate(files):
        with open(file, "r") as f:
            sma, ecc = map(lambda x: float(x), f.read().split(","))
            semimajor_axes[i] = sma
            eccentricities[i] = ecc

    print(eccentricities)
    print(semimajor_axes)
    hist = np.zeros((RESOLUTION_1D, RESOLUTION_1D))
    ecc_indices = np.digitize(eccentricities, np.linspace(*EXTENT[:2], RESOLUTION_1D))
    sma_indices = np.digitize(semimajor_axes, np.linspace(*EXTENT[2:], RESOLUTION_1D))

    for i in range(len(semimajor_axes)):
        hist[ecc_indices[i] - 1, sma_indices[i] - 1] += 1

    fig, ax = plt.subplots()
    # plt.plot(eccentricities, semimajor_axes, "g.")
    pic = ax.imshow(
        hist[:, ::-1].T,
        extent=EXTENT,
        interpolation="nearest",
        aspect="auto",
        cmap="gray_r",
    )
    ax.set_xlabel("Eccentricity", fontsize=mnras.FONT_SIZE)
    ax.set_ylabel("Semi-major axis, kpc", fontsize=mnras.FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)
    cbar = plt.colorbar(pic)
    cbar.set_label("Number of models with given orbit", fontsize=mnras.FONT_SIZE)
    cbar.ax.tick_params(labelsize=mnras.FONT_SIZE)

    plt.show()
