import glob
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

from scriptslib import mnras, plot as splot


RESULTS_DIR = "example_velocity/results/{}"


def prepare_axes(dist_axes, bound_mass_axes):
    for ax in dist_axes, bound_mass_axes:
        ax.grid(True)
        ax.set_ylim(0)
        ax.set_xlim(0, 4)
        ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    # sort labels in axes
    handles, labels = dist_axes.get_legend_handles_labels()
    dist_axes.legend(handles, labels)
    dist_axes.set_ylabel("Distance, kpc", fontsize=mnras.FONT_SIZE)

    bound_mass_axes.set_xlabel("Time, Gyr", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.set_ylabel("Bound mass, $10^{11}$ MSun", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def show():
    filenames = glob.glob(RESULTS_DIR.format("*.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(mnras.size_from_aspect(1))
    fig.subplots_adjust(wspace=0, hspace=0)

    for i, filename in enumerate(filenames):
        angle = float(Path(filename).stem)
        parameters = pd.read_csv(filename)
        parameters.bound_mass = parameters.bound_mass
        max_bound_mass = parameters.bound_mass.to_numpy()[0]

        threshold = np.argmax(parameters.bound_mass < 0.01 * max_bound_mass)

        ax1.plot(
            parameters["times"][:threshold],
            parameters["distances"][:threshold],
            label=f"{angle}",
            color=splot.colors[2 * i],
        )
        ax2.plot(
            parameters["times"],
            parameters["bound_mass"],
            label=f"{angle}",
            color=splot.colors[2 * i],
        )

    prepare_axes(ax1, ax2)

    plt.savefig(RESULTS_DIR.format("velocity_vector.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    show()
