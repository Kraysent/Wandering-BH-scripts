from amuse.lab import Particles, units
import pandas as pd
import numpy as np
import pyfalcon
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from datetime import datetime

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

EPS = 0.5 | units.kpc
DT = 0.5**6
MAX_TIME = 5.0

RESULTS_DIR = "models_resolution/results/{}"


def _read_csv(path: str) -> Particles:
    df = pd.read_csv(path, delimiter=" ")

    particles = Particles(len(df))
    particles.x = np.array(df.x) | SPACE_UNIT
    particles.y = np.array(df.y) | SPACE_UNIT
    particles.z = np.array(df.z) | SPACE_UNIT
    particles.vx = np.array(df.vx) | VEL_UNIT
    particles.vy = np.array(df.vy) | VEL_UNIT
    particles.vz = np.array(df.vz) | VEL_UNIT
    particles.mass = np.array(df.m) | MASS_UNIT

    return particles


def _downsample(particles: Particles, to: int) -> Particles:
    coeff = len(particles) / to
    subset_indices = np.random.choice(len(particles), to, replace=False)
    new_particles = particles[subset_indices]
    new_particles.mass = new_particles.mass * coeff

    return new_particles


def _leapfrog(particles: Particles):
    acceleration, _ = pyfalcon.gravity(
        particles.position.value_in(SPACE_UNIT),
        particles.mass.value_in(MASS_UNIT),
        EPS.value_in(SPACE_UNIT),
    )

    particles.velocity += acceleration * DT / 2 | VEL_UNIT
    particles.position += particles.velocity * (DT | TIME_UNIT)

    acceleration, _ = pyfalcon.gravity(
        particles.position.value_in(SPACE_UNIT),
        particles.mass.value_in(MASS_UNIT),
        EPS.value_in(SPACE_UNIT),
    )

    particles.velocity += acceleration * DT / 2 | VEL_UNIT

    return particles


def _prepare_axes(dist_axes, bound_mass_axes):
    for ax in dist_axes, bound_mass_axes:
        ax.grid(True)
        ax.set_ylim(0)
        ax.legend()
        ax.set_xlabel("Time, Gyr")

    dist_axes.set_title("Distance between centres of mass of two galaxies\nfor different number of particles")
    dist_axes.set_ylabel("Distance, kpc")

    bound_mass_axes.set_title("Bound mass of the satellite for different number of particles")
    bound_mass_axes.set_ylabel("Bound mass, 232500 * MSun")
    bound_mass_axes.ticklabel_format(axis='y', style="sci", scilimits=(0,0))

def model(save_trajectories: bool = False, save: bool = False):
    samples = [
        (5000, 2500),
        (20000, 10000),
        (200000, 100000),
        (400000, 200000),
        (600000, 300000),
        (1000000, 500000),
    ]

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    vector_length = lambda v, unit, axis = 0: (v.value_in(unit) ** 2).sum(axis = axis) ** 0.5

    def distance(host: Particles, satellite: Particles) -> float:
        return vector_length(host.center_of_mass() - satellite.center_of_mass(), SPACE_UNIT)

    def bound_mass(galaxy: Particles) -> float:
        NUMBER_OF_ITERATIONS = 15
        CHANGE_THRESHOLD = 0.01
        change = 1.0
        curr_len, prev_len = len(galaxy), 0
        i = 0

        while change >= CHANGE_THRESHOLD:
            _, potentials = pyfalcon.gravity(
                galaxy.position.value_in(SPACE_UNIT),
                galaxy.mass.value_in(MASS_UNIT),
                EPS.value_in(SPACE_UNIT),
            )
            velocities = vector_length(galaxy.velocity - galaxy.center_of_mass_velocity(), VEL_UNIT, axis=1)
            full_specific_energies = potentials + velocities**2 / 2

            galaxy = galaxy[full_specific_energies < 0]

            prev_len = curr_len
            curr_len = len(galaxy)

            if curr_len == 0 or prev_len == 0:
                break

            change = (prev_len - curr_len) / prev_len
            i += 1

            if i >= NUMBER_OF_ITERATIONS:
                break


        return galaxy.total_mass().value_in(MASS_UNIT)

    for host_sample, sat_sample in samples:
        host_particles = _downsample(
            _read_csv("models_resolution/models/host.csv"), host_sample
        )
        sat_particles = _downsample(
            _read_csv("models_resolution/models/sat.csv"), sat_sample
        )

        sat_particles.position += [100, 0, 0] | units.kpc
        sat_particles.velocity += [0, 180, 0] | units.kms

        particles = host_particles
        particles.add_particles(sat_particles)

        # Create dataframe to store all of the parameters that we want to track while evolving the system.
        parameters = pd.DataFrame()
        parameters["times"] = np.arange(0, MAX_TIME, DT)
        parameters["distances"] = [0] * len(parameters)
        parameters["bound_mass"] = [0] * len(parameters)

        for i in parameters.index:
            particles = _leapfrog(particles)

            parameters.at[i, "distances"] = distance(particles[:host_sample], particles[-sat_sample:])
            parameters.at[i, "bound_mass"] = bound_mass(particles[-sat_sample:])

            print(
                f"{datetime.now().strftime('%H:%M')}\t{parameters.times[i]:.02f}\t{parameters.distances[i]:.02f}\t{parameters.bound_mass[i]:.02f}"
            )

        ax1.plot(
            parameters.times, parameters.distances, label=f"{host_sample + sat_sample}"
        )
        ax2.plot(
            parameters.times, parameters.bound_mass, label=f"{host_sample + sat_sample}"
        )

        if save_trajectories:
            parameters.to_csv(RESULTS_DIR.format(f"{host_sample+sat_sample}.csv"), index=False)

    _prepare_axes(ax1, ax2)

    if save:
        fig1.savefig(RESULTS_DIR.format("distance.pdf"))
        fig2.savefig(RESULTS_DIR.format("bound_mass.pdf"))
    else:
        plt.show()


def load(save: str | None = None):
    filenames = glob.glob(RESULTS_DIR.format("*.csv"))

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for filename in filenames:
        print(f"Reading {filename}")
        number_of_particles = int(Path(filename).stem)
        parameters = pd.read_csv(filename, index_col=None)
        ax1.plot(
            parameters.times, parameters.distances, label=f"{number_of_particles}"
        )
        ax2.plot(
            parameters.times, parameters.bound_mass, label=f"{number_of_particles}"
        )

    _prepare_axes(ax1, ax2)

    if save:
        fig1.savefig(RESULTS_DIR.format("distance.pdf"))
        fig2.savefig(RESULTS_DIR.format("bound_mass.pdf"))
    else:
        plt.show()
