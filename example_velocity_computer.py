from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd
from amuse.lab import Particles, ScalarQuantity, units

import scriptslib
from scriptslib import particles as sparticles
from scriptslib import physics


@dataclass
class Parameters:
    angle: float


params_list = [
    Parameters(0),
    Parameters(45),
    Parameters(90),
]

MODELS_DIR = "example_velocity/models/{}"
RESULTS_DIR = "example_velocity/results/{}"
HOST_N = 1000000
SAT_N = 500000
INCLINATION = np.deg2rad(30)
DISTANCE_ABS = 100
VEL_ABS = 180
DT = 0.5**6 | units.Gyr
EPS = 0.4 | units.kpc
SAT_MASS = 2e11 | units.MSun
MAX_TIME = 5.0 | units.Gyr
MASS_THRESHOLD = 0.05


def prepare_particles(params: Parameters):
    host_particles = sparticles.pipe(
        scriptslib.read_csv(MODELS_DIR.format("host.csv")),
        sparticles.downsample(HOST_N),
        sparticles.set_attribute("subset", "host"),
    )

    angle = np.deg2rad(params.angle)
    sat_particles = sparticles.pipe(
        scriptslib.read_csv(MODELS_DIR.format("sat.csv")),
        sparticles.downsample(SAT_N),
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


def process(param: Parameters):
    particles = prepare_particles(param)

    parameters = pd.DataFrame()
    parameters["times"] = np.arange(0, MAX_TIME.value_in(units.Gyr), DT.value_in(units.Gyr))
    parameters["distances"] = [0] * len(parameters)
    parameters["bound_mass"] = [0] * len(parameters)

    print(f"Started simulation {param}")
    time = 0 | units.Gyr

    for i, particles in enumerate(evolve(particles, EPS, DT)):
        parameters.at[i, "distances"] = physics.distance(
            particles.select(lambda subset: subset == "host", ["subset"]),
            particles.select(lambda subset: subset == "sat", ["subset"]),
        )
        bound_subset = physics.bound_subset(particles.select(lambda subset: subset == "sat", ["subset"]), EPS)
        parameters.at[i, "bound_mass"] = bound_subset.total_mass().value_in(units.MSun)
        disruption = bound_subset.total_mass() / SAT_MASS
        print(
            f"{param.angle}\t{datetime.now().strftime('%H:%M:%S')}\t{time.value_in(units.Gyr):.03f}\t{disruption:.03f}"
        )

        if disruption <= MASS_THRESHOLD:
            break

        time += DT

        if time >= MAX_TIME:
            break

    parameters.to_csv(RESULTS_DIR.format(f"{param.angle}.csv"))


if __name__ == "__main__":
    with futures.ThreadPoolExecutor(max_workers=9) as executor:
        executor.map(process, params_list)

    # for param in params_list:
    # process(param)
