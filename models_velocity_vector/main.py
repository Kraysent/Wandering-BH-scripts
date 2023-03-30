from collections import namedtuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from amuse.lab import units

import scriptslib
from scriptslib import mnras, physics

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

EPS = 0.2 | units.kpc
DT = 0.5**6
MAX_TIME = 5.0
HOST_SAMPLE = 400000
SAT_SAMPLE = 200000
VEL_ABS = 180

RESULTS_DIR = "models_velocity_vector/results/{}"
MODELS_DIR = "models_velocity_vector/models/{}"

params = [(0, "r"), (45, "g"), (90, "b")]

settings = namedtuple("settings", ["figaspect", "scale"])

modes_settings = {
    "paper": settings(1, 1),
    "presentation": settings(0.6, 1.5)
}

def compute():
    for angle, _ in params:
        angle = np.deg2rad(angle)
        host_particles = scriptslib.downsample(
            scriptslib.read_csv(MODELS_DIR.format("host.csv"), SPACE_UNIT, VEL_UNIT, MASS_UNIT),
            HOST_SAMPLE,
        )
        sat_particles = scriptslib.downsample(
            scriptslib.read_csv(MODELS_DIR.format("sat.csv"), SPACE_UNIT, VEL_UNIT, MASS_UNIT),
            SAT_SAMPLE,
        )
        sat_particles.position += [100, 0, 0] | units.kpc
        sat_particles.velocity += [
            -np.cos(angle) * VEL_ABS,
            np.sin(angle) * VEL_ABS,
            0,
        ] | VEL_UNIT

        particles = host_particles
        particles.add_particles(sat_particles)

        parameters = pd.DataFrame()
        parameters["times"] = np.arange(0, MAX_TIME, DT)
        parameters["distances"] = [0] * len(parameters)
        parameters["bound_mass"] = [0] * len(parameters)

        for i in parameters.index:
            particles = physics.leapfrog(
                particles,
                EPS,
                DT | TIME_UNIT,
                SPACE_UNIT,
                VEL_UNIT,
                MASS_UNIT,
                TIME_UNIT,
            )

            bound_subset = physics.bound_subset(particles[-SAT_SAMPLE:], EPS, SPACE_UNIT, MASS_UNIT, VEL_UNIT)
            parameters.at[i, "distances"] = physics.distance(
                particles[:HOST_SAMPLE],
                bound_subset,
                SPACE_UNIT,
            )
            parameters.at[i, "bound_mass"] = bound_subset.total_mass().value_in(units.MSun)

            print(
                f"{datetime.now().strftime('%H:%M')}\t{parameters.times[i]:.02f}\t{parameters.distances[i]:.02f}\t{parameters.bound_mass[i]:.02f}"
            )

        parameters.to_csv(RESULTS_DIR.format(f"{np.rad2deg(angle):.00f}.csv"), index=False)


def _prepare_axes(dist_axes, bound_mass_axes):
    for ax in dist_axes, bound_mass_axes:
        ax.grid(True)
        ax.set_xlim(0, 4.2)
        ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    dist_axes.legend(prop={"size": mnras.FONT_SIZE})
    dist_axes.set_ylabel("Distance, kpc", fontsize=mnras.FONT_SIZE)
    dist_axes.set_ylim(0, 130)

    bound_mass_axes.set_xlabel("Time, Gyr", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.set_ylabel("Bound mass, $10^{11}$ MSun", fontsize=mnras.FONT_SIZE)
    bound_mass_axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    bound_mass_axes.set_ylim(0, 2.4)


def _prepare_figure(fig, mode: settings):
    fig.set_size_inches(mnras.size_from_aspect(mode.figaspect, scale=mode.scale))
    fig.subplots_adjust(wspace=0, hspace=0)

def plot(save: bool, mode: str):
    mode = modes_settings[mode]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    _prepare_figure(fig, mode)

    for angle, color in params:
        filename = RESULTS_DIR.format(f"{angle}.csv")
        print(f"Reading {filename}")
        parameters = pd.read_csv(filename, index_col=None)
        parameters.bound_mass = parameters.bound_mass
        max_bound_mass = parameters.bound_mass.to_numpy()[0]

        threshold = np.argmax(parameters.bound_mass < 0.01 * max_bound_mass)
        ax1.plot(
            parameters.times[:threshold],
            parameters.distances[:threshold],
            label=f"${angle}^\circ$",
            color=color,
        )
        ax2.plot(
            parameters.times,
            parameters.bound_mass / 1e11,
            label=f"{angle}",
            color=color,
        )

    _prepare_axes(ax1, ax2)

    if save:
        fig.savefig(RESULTS_DIR.format("result.pdf"), pad_inches=0, bbox_inches="tight")
    else:
        plt.show()
