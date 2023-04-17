from datetime import datetime

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from amuse.lab import units
from matplotlib import figure
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import scriptslib
from scriptslib import mnras, physics
from scriptslib import plot as splot

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

MAX_TIME = 13.7  # Gyr
DT = 0.5**6  # Gyr
DT_UPDATE = 0.5**3  # Gyr
EPS = 0.5 | units.kpc
PLOT_ITERATION = int(DT_UPDATE / DT)
HOST_N = 400000
SAT_N = 200000

RESULTS_DIR = "models_example/results/{}"
EXTENT = [-100, 100, -100, 100]
RESOLUTION = 500


def _prepare_axes(ax):
    ax.set_xlabel("x, kpc")
    ax.set_ylabel("y, kpc")
    ax.legend(
        handles=[
            Patch(facecolor="r", edgecolor="r", label="Host"),
            Patch(facecolor="b", edgecolor="b", label="Satellite"),
        ]
    )


def model(save: bool, plot: bool):
    fig, ax = plt.subplots()
    _prepare_axes(ax)
    plt.tight_layout()

    if not save:
        plt.ion()
        plt.show()

    host_particles = scriptslib.downsample(
        scriptslib.read_csv("models_resolution/models/host.csv", SPACE_UNIT, VEL_UNIT, MASS_UNIT),
        HOST_N,
    )
    sat_particles = scriptslib.downsample(
        scriptslib.read_csv("models_resolution/models/sat.csv", SPACE_UNIT, VEL_UNIT, MASS_UNIT),
        SAT_N,
    )

    sat_particles.position += [100, 0, 0] | units.kpc
    sat_particles.velocity += [0, 180, 0] | units.kms

    particles = host_particles
    particles.add_particles(sat_particles)
    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    time = 0
    i = 0

    while time < MAX_TIME:
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{time:.02f}")
        particles = physics.leapfrog(
            particles,
            EPS,
            DT | TIME_UNIT,
            SPACE_UNIT,
            VEL_UNIT,
            MASS_UNIT,
            TIME_UNIT,
        )

        if i % PLOT_ITERATION == 0:
            # first 20 percent of each subset is barion matter so plotting only it
            sat_bound = physics.bound_subset(particles[-SAT_N:], EPS, SPACE_UNIT, MASS_UNIT, VEL_UNIT)
            host_barion, sat_barion = (
                particles[: int(0.2 * HOST_N)],
                particles[-SAT_N : -int(SAT_N * 0.8)],
            )

            _, rgb_map = splot.plot_hist(
                red_x=host_barion.x.value_in(SPACE_UNIT),
                red_y=host_barion.y.value_in(SPACE_UNIT),
                blue_x=sat_barion.x.value_in(SPACE_UNIT),
                blue_y=sat_barion.y.value_in(SPACE_UNIT),
                green_x=sat_bound.x.value_in(SPACE_UNIT),
                green_y=sat_bound.y.value_in(SPACE_UNIT),
                extent=EXTENT,
                resolution=RESOLUTION,
                axes=ax,
                return_rgbmap=True,
            )
            ax.set_title(f"{time:.02f} Gyr")

            if save:
                np.save(RESULTS_DIR.format(f"bins/{time:.02f}.npy"), rgb_map)
                fig.savefig(RESULTS_DIR.format(f"pdfs/{time:.02f}.pdf"))
            else:
                fig.canvas.draw()
                plt.pause(1e-3)

        i += 1
        time += DT


def _prepare_axes_result(ax, time: float):
    ax.set_title(f"{time:.02f} Gyr", y=1.0, pad=-14, fontsize=mnras.FONT_SIZE)
    ax.set_box_aspect(1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])


def plot_separate_pic(save: bool):
    # using ndarray to ensure that this is matrix and not list of lists
    times = np.array(
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.25,
            2.5,
            2.75,
            3.0,
        ]
    )

    for i, time in enumerate(times):
        fig, ax = plt.subplots()
        plt.tight_layout()
        fig.set_size_inches(mnras.size_from_aspect(1))

        _prepare_axes_result(ax, time)
        ax.legend(
            handles=[
                Patch(facecolor="r", edgecolor="r", label="Host"),
                Patch(facecolor="b", edgecolor="b", label="Satellite"),
            ],
            prop={"size": mnras.FONT_SIZE},
            loc="upper right",
        )

        scalebar = AnchoredSizeBar(
            ax.transData,
            50,
            "50 kpc",
            "lower right",
            pad=0.4,
            color="black",
            frameon=False,
            size_vertical=1,
            fontproperties=fm.FontProperties(size=mnras.FONT_SIZE),
        )
        ax.add_artist(scalebar)

        rgb_map = np.load(RESULTS_DIR.format(f"bins/{time:.02f}.npy"))
        ax.imshow(rgb_map, extent=EXTENT, interpolation="nearest", aspect="auto")

        if save:
            fig.savefig(RESULTS_DIR.format(f"{i+1}.pdf"), bbox_inches="tight", pad_inches=0)
        else:
            plt.show()

        plt.close(fig)


def plot_plane(save: bool):
    # using ndarray to ensure that this is matrix and not list of lists
    times = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.75],
            [2.5, 3.25],
        ]
    )

    fig, axes = plt.subplots(*times.shape, sharex="all", sharey="all")

    # prepare axes
    plt.tight_layout()
    # fig.set_size_inches(figure.figaspect(3 / 2) * 2)
    fig.set_size_inches(mnras.size_from_aspect(3 / 2))
    fig.subplots_adjust(wspace=0, hspace=0)

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            _prepare_axes_result(ax, times[i, j])

    axes[0, 0].legend(
        handles=[
            Patch(facecolor="r", edgecolor="r", label="Host"),
            Patch(facecolor="b", edgecolor="b", label="Satellite"),
        ],
        prop={"size": mnras.FONT_SIZE},
        loc="lower right",
    )

    scalebar = AnchoredSizeBar(
        axes[0, 0].transData,
        50,
        "50 kpc",
        "lower right",
        pad=0.4,
        color="black",
        frameon=False,
        size_vertical=1,
        fontproperties=fm.FontProperties(size=mnras.FONT_SIZE),
    )

    axes[0, 1].add_artist(scalebar)

    for i in range(times.shape[0]):
        for j in range(times.shape[1]):
            bin_filename = RESULTS_DIR.format(f"bins/{times[i, j]:.02f}.npy")
            rgb_map = np.load(bin_filename)
            axes[i, j].imshow(rgb_map, extent=EXTENT, interpolation="nearest", aspect="auto")

    if save:
        plt.savefig(RESULTS_DIR.format("result.pdf"), pad_inches=0, bbox_inches="tight")
    else:
        plt.show()
