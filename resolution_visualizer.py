import glob
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

from scriptslib import mnras, plot as splot


RESULTS_DIR = "resolution/results/{}"


def prepare_axes(dist_axes, bound_mass_axes):
    for ax in dist_axes, bound_mass_axes:
        ax.grid(True)
        ax.set_xlim(0, 5)
        ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    # sort labels in axes
    handles, labels = dist_axes.get_legend_handles_labels()
    order = np.argsort([int(l) for l in labels])
    dist_axes.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        prop={"size": mnras.FONT_SIZE},
    )
    dist_axes.set_ylabel("Distance, kpc", fontsize=mnras.FONT_SIZE)

    bound_mass_axes.set_xlabel("Time, Gyr", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.set_ylabel("Bound mass, $10^{11}$ MSun", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def show():
    filenames = glob.glob(RESULTS_DIR.format("*-*.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(mnras.size_from_aspect(1))
    fig.subplots_adjust(wspace=0, hspace=0)
    prepare_axes(ax1, ax2)
    plt.tight_layout()

    for i, filename in enumerate(filenames):
        host_n, sat_n = map(lambda x: int(x), Path(filename).stem.split("-"))
        parameters = pd.read_csv(filename)
        ax1.plot(
            parameters["times"],
            parameters["distances"],
            label=f"{host_n+sat_n}",
            color=splot.colors[i],
        )
        ax2.plot(
            parameters["times"],
            parameters["bound_mass"],
            label=f"{host_n+sat_n}",
            color=splot.colors[i],
        )

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.legend(prop={"size": mnras.FONT_SIZE})
    ax2.legend(prop={"size": mnras.FONT_SIZE})

    plt.savefig(RESULTS_DIR.format("resolution.pdf"))


if __name__ == "__main__":
    show()
