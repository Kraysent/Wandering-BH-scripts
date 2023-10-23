from collections import deque, namedtuple
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from amuse.lab import Particles, ScalarQuantity, units

import scriptslib
from scriptslib import ellipse_approx, mnras
from scriptslib import particles as sparticles
from scriptslib import physics

EPS = 0.4 | units.kpc
DT = 0.5**6 | units.Gyr
HOST_SAMPLE = 400000
SAT_SAMPLE = 200000
VEL_ABS = 180  # kms
DISTANCE_ABS = 100  # kpc
INCLINATION = np.deg2rad(30)
MASS_TRESHOLD = 0.05
SAT_MASS = 5e11 | units.MSun
ORBIT_QUEUE_LENGTH = 30

RESULTS_DIR = "models_velocity_vector/results/{}"
MODELS_DIR = "models_velocity_vector/models/{}"


@dataclass
class Params:
    speed_angle: float
    line_clr: str
    name: str
    sat_model: str


Settings = namedtuple("Settings", ["figaspect", "scale"])

modes_settings = {"paper": Settings(1, 1), "presentation": Settings(0.6, 1.5)}

rng = range(0, 50)
params = (
    []
    + [Params(35, "r", f"i30e35_{i}", "sat2.hdf5") for i in rng]
    + [Params(40, "r", f"i30e40_{i}", "sat2.hdf5") for i in rng]
    + [Params(45, "r", f"i30e45_{i}", "sat2.hdf5") for i in rng]
    + [Params(50, "r", f"i30e50_{i}", "sat2.hdf5") for i in rng]
    + [Params(55, "r", f"i30e55_{i}", "sat2.hdf5") for i in rng]
)


def prepare_particle_set(params: Params):
    """
    Setup: host is initially located in the origin without any rotation, satellite is located
    on distance `DISTANCE_ABS` with inclination `INCLINATION` in such way that its centre has
    only X and Z components, but no Y.

    The satellite is rotated around Y axis on angle `INCLINATION` so its plane includes the line
    from host centre to its centre.
    """
    host_particles = sparticles.pipe(
        scriptslib.read_csv(MODELS_DIR.format("host.csv")),
        sparticles.downsample(HOST_SAMPLE),
        sparticles.set_attribute("subset", "host"),
    )

    angle = np.deg2rad(params.speed_angle)
    sat_particles = sparticles.pipe(
        scriptslib.read_hdf5(MODELS_DIR.format(params.sat_model)),
        sparticles.downsample(SAT_SAMPLE),
        sparticles.rotate("y", INCLINATION),
        sparticles.append_position([np.cos(INCLINATION), 0, np.sin(INCLINATION)] * (DISTANCE_ABS | units.kpc)),
        sparticles.append_velocity(
            [
                -np.cos(angle) * np.cos(INCLINATION),
                np.sin(angle),
                -np.cos(angle) * np.sin(INCLINATION),
            ]
            * (VEL_ABS | units.kms)
        ),
        sparticles.set_attribute("subset", "sat"),
    )

    return sparticles.pipe(
        Particles(),
        sparticles.append_particles(host_particles),
        sparticles.append_particles(sat_particles),
        sparticles.move_to_origin(),
        sparticles.enumerate(),
    )


def evolve(
    particles: Particles,
    eps: ScalarQuantity,
    dt: ScalarQuantity,
) -> Iterator[Particles]:
    while True:
        particles = physics.leapfrog(particles, eps, dt)
        yield particles


def process(param: Params):
    print(f"Generating particles for {param}")
    particles = prepare_particle_set(param)
    history = deque()
    total_mass = particles.select(lambda subset: subset == "sat", ["subset"]).total_mass()

    print(f"Started simulation {param}")
    time = 0 | units.Gyr

    for particles in evolve(particles, EPS, DT):
        time += DT
        history.append(particles.copy())

        if len(history) > ORBIT_QUEUE_LENGTH:
            history.popleft()

        bound_subset = physics.bound_subset(particles.select(lambda subset: subset == "sat", ["subset"]), EPS)
        disruption = bound_subset.total_mass() / total_mass
        print(
            "\t".join(
                [
                    param.name,
                    datetime.now().strftime("%H:%M:%S"),
                    f"{time.value_in(units.Gyr):.03f}",
                    f"{disruption:.03f}",
                    str(len(history)),
                ]
            )
        )

        if disruption <= MASS_TRESHOLD:
            indexes = bound_subset.id
            positions = np.zeros((ORBIT_QUEUE_LENGTH, 3))

            for s_index, snapshot in enumerate(history):
                positions[s_index, :] = snapshot[indexes].center_of_mass().value_in(units.kpc)

            semimajor_axis, eccentricity = ellipse_approx.fit_3d_ellipse(positions)

            with open(RESULTS_DIR.format(f"{param.name}.csv"), "w") as file:
                file.writelines([f"{semimajor_axis},{eccentricity}"])

            break


if __name__ == "__main__":
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(process, params)
