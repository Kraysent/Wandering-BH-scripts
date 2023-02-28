from datetime import datetime
from amuse.lab import units
import scriptslib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

MAX_TIME = 13.7
DT = 0.5**6
DT_UPDATE = 0.5**2
EPS = 0.5 | units.kpc
PLOT_ITERATION = int(DT_UPDATE / DT)
HOST_N = 400000
SAT_N = 200000

RESULTS_DIR = "models_example/results/{}"
EXTENT = [-100, 100, -100, 100]
RESOLUTION = 500


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


def _prepare_axes(ax):
    ax.set_xlabel("x, kpc")
    ax.set_ylabel("y, kpc")
    ax.legend(
        handles=[
            Patch(facecolor="r", edgecolor="r", label="Host"),
            Patch(facecolor="b", edgecolor="b", label="Satellite"),
        ]
    )


def model(save: bool):
    fig, ax = plt.subplots()
    _prepare_axes(ax)
    plt.tight_layout()

    if not save:
        plt.ion()
        plt.show()

    host_particles = scriptslib.downsample(
        scriptslib.read_csv("models_resolution/models/host.csv", SPACE_UNIT, VEL_UNIT, MASS_UNIT), HOST_N
    )
    sat_particles = scriptslib.downsample(
        scriptslib.read_csv("models_resolution/models/sat.csv", SPACE_UNIT, VEL_UNIT, MASS_UNIT), SAT_N
    )

    sat_particles.position += [100, 0, 0] | units.kpc
    sat_particles.velocity += [-180/1.41, 180/1.41, 0] | units.kms

    particles = host_particles
    particles.add_particles(sat_particles)
    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    time = 0
    i = 0

    while time < MAX_TIME:
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{time:.02f}")
        particles = scriptslib.leapfrog(particles, EPS, DT | TIME_UNIT, SPACE_UNIT, VEL_UNIT, MASS_UNIT, TIME_UNIT)

        if i % PLOT_ITERATION == 0:
            # first 20 percent of each subset is barion matter so plotting only it
            host, sat = particles[: int(0.2 * HOST_N)], particles[-SAT_N : -int(SAT_N * 0.8)]

            host_hist, _, _ = np.histogram2d(
                host.x.value_in(SPACE_UNIT), host.y.value_in(SPACE_UNIT), RESOLUTION, [EXTENT[:2], EXTENT[2:]]
            )
            sat_hist, _, _ = np.histogram2d(
                sat.x.value_in(SPACE_UNIT), sat.y.value_in(SPACE_UNIT), RESOLUTION, [EXTENT[:2], EXTENT[2:]]
            )

            host_hist = _log_scale(host_hist.T[::-1, :], low=0.4)
            sat_hist = _log_scale(sat_hist.T[::-1, :], low=0.4)

            rgb_map = np.stack([host_hist, np.zeros(host_hist.shape), sat_hist], axis=2)
            rgb_map[(rgb_map[:, :] ** 2).sum(axis=2) == 0] = 0.85

            ax.imshow(rgb_map, extent=EXTENT, interpolation="nearest", aspect="auto")
            ax.set_title(f"{time:.02f} Gyr")

            if save:
                fig.savefig(RESULTS_DIR.format(f"{time:.02f}.pdf"))
            else:
                fig.canvas.draw()
                plt.pause(1e-3)

        i += 1
        time += DT
