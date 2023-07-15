from collections import deque, namedtuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from amuse.lab import units

import scriptslib
import scriptslib.plot as splot
from scriptslib import ellipse_approx, mnras, physics, rotate

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

EPS = 0.2 | units.kpc
DT = 0.5**6
MAX_TIME = 5.0
HOST_SAMPLE = 400000
SAT_SAMPLE = 200000
VEL_ABS = 180  # kms
DISTANCE_ABS = 100  # kpc
INCLINATION = np.deg2rad(30)
MASS_TRESHOLD = 0.05
SAT_MASS = 2e11 | units.MSun
ORBIT_QUEUE_LENGTH = 30

RESULTS_DIR = "models_velocity_vector/results/{}"
MODELS_DIR = "models_velocity_vector/models/{}"

Params = namedtuple("Params", ["speed_angle", "line_clr", "name"])
Settings = namedtuple("Settings", ["figaspect", "scale"])

modes_settings = {"paper": Settings(1, 1), "presentation": Settings(0.6, 1.5)}

rng = range(180, 210)
params = (
    # [Params(35, "r", f"i30e35_{i}") for i in rng]
    [Params(40, "r", f"i30e40_{i}") for i in rng]
    # + [Params(45, "r", f"i30e45_{i}") for i in rng]
    + [Params(50, "r", f"i30e50_{i}") for i in rng]
    # + [Params(55, "r", f"i30e55_{i}") for i in rng]
)


def prepare_model(params: Params):
    """
    Setup: host is initially located in the origin without any rotation, satellite is located
    on distance `DISTANCE_ABS` with inclination `INCLINATION` in such way that its centre has
    only X and Z components, but no Y.

    The satellite is rotated around Y axis on angle `INCLINATION` so its plane includes the line
    from host to itself centres.
    """
    host_particles = scriptslib.downsample(scriptslib.read_csv(MODELS_DIR.format("host.csv")), HOST_SAMPLE)
    sat_particles = scriptslib.downsample(scriptslib.read_csv(MODELS_DIR.format("sat.csv")), SAT_SAMPLE)
    angle = np.deg2rad(params.speed_angle)
    print("Read models")

    rotate(sat_particles, "y", INCLINATION)
    print("Rotated models")

    sat_particles.position += [
        DISTANCE_ABS * np.cos(INCLINATION),
        0,
        DISTANCE_ABS * np.sin(INCLINATION),
    ] | units.kpc
    sat_particles.velocity += [
        -VEL_ABS * np.cos(angle) * np.cos(INCLINATION),
        VEL_ABS * np.sin(angle),
        -VEL_ABS * np.cos(angle) * np.sin(INCLINATION),
    ] | VEL_UNIT
    particles = host_particles
    particles.add_particles(sat_particles)

    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    particles.id = np.arange(0, len(particles))

    return particles


def compute(debug: bool = False):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.tight_layout()
    fig.set_size_inches(mnras.size_from_aspect(1 / 2))
    fig.subplots_adjust(wspace=0, hspace=0)

    for param in params:
        particles = prepare_model(param)
        history = deque()

        print(f"Started simulation {param}")
        times = np.arange(0, MAX_TIME, DT)

        for i, time in enumerate(times):
            particles = physics.leapfrog(particles, EPS, DT | TIME_UNIT)
            history.append(particles.copy())

            if len(history) > ORBIT_QUEUE_LENGTH:
                history.popleft()

            bound_subset = physics.bound_subset(particles[-SAT_SAMPLE:], EPS)
            print(f"{datetime.now().strftime('%H:%M:%S')}\t{time:.03f}\t{bound_subset.total_mass() / SAT_MASS:.03f}")

            if bound_subset.total_mass() / SAT_MASS <= MASS_TRESHOLD:
                indexes = bound_subset.id
                positions = np.zeros((ORBIT_QUEUE_LENGTH, 3))

                for s_index, snapshot in enumerate(history):
                    positions[s_index, :] = snapshot[indexes].center_of_mass().value_in(units.kpc)

                semimajor_axis, eccentricity = ellipse_approx.fit_3d_ellipse(positions)
                # ax = plt.figure().add_subplot(projection="3d")
                # ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
                # ax.set_xlabel(f"{semimajor_axis:.04f}\t{eccentricity:.04f}")
                # plt.show()

                with open(RESULTS_DIR.format(f"{param.name}.csv"), "w") as file:
                    file.writelines([f"{semimajor_axis},{eccentricity}"])

                break

            if debug and (i % 10 == 0):
                splot.plot_hist(
                    particles.x.value_in(units.kpc),
                    particles.y.value_in(units.kpc),
                    extent=[-100, 100, -100, 100],
                    axes=ax1,
                )
                splot.plot_hist(
                    particles.z.value_in(units.kpc),
                    particles.y.value_in(units.kpc),
                    extent=[-100, 100, -100, 100],
                    axes=ax2,
                )
                plt.savefig(RESULTS_DIR.format(f"debug/{i}.png"))
                ax1.clear()
                ax2.clear()


def _prepare_axes(dist_axes, bound_mass_axes):
    for ax in dist_axes, bound_mass_axes:
        ax.grid(True)
        ax.set_xlim(0, 4.2)
        ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    dist_axes.legend(prop={"size": mnras.FONT_SIZE})
    dist_axes.set_ylabel("Distance, kpc", fontsize=mnras.FONT_SIZE)
    dist_axes.set_ylim(0, 160)

    bound_mass_axes.set_xlabel("Time, Gyr", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.set_ylabel("Bound mass, $10^{11}$ MSun", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    bound_mass_axes.set_ylim(0, 2.4)


def _prepare_figure(fig, mode: Settings):
    fig.set_size_inches(mnras.size_from_aspect(mode.figaspect, scale=mode.scale))
    fig.subplots_adjust(wspace=0, hspace=0)


def plot(save: bool, mode: str):
    mode = modes_settings[mode]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    _prepare_figure(fig, mode)

    for param in params:
        filename = RESULTS_DIR.format(f"{param.name}.csv")
        print(f"Reading {filename}")
        parameters = pd.read_csv(filename, index_col=None)
        parameters.bound_mass = parameters.bound_mass
        distances = (parameters.x**2 + parameters.y**2 + parameters.z**2) ** 0.5
        max_bound_mass = parameters.bound_mass.to_numpy()[0]

        threshold = np.argmax(parameters.bound_mass < 0.01 * max_bound_mass)
        ax1.plot(
            parameters.times[:threshold],
            distances[:threshold],
            label=f"{param.name}",
            color=param.line_clr,
        )
        ax2.plot(
            parameters.times,
            parameters.bound_mass / 1e11,
            label=f"{param.name}",
            color=param.line_clr,
        )

    _prepare_axes(ax1, ax2)

    if save:
        fig.savefig(RESULTS_DIR.format("result.pdf"), pad_inches=0, bbox_inches="tight")
    else:
        plt.show()
