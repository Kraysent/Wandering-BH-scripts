from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from amuse.lab import units, Particles
from collections import deque
import pandas as pd

import numpy as np

import scriptslib
import matplotlib.pyplot as plt
from scriptslib import physics, ellipse_approx

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

MAX_TIME = 5  # Gyr
DT = 0.5**8  # Gyr
DT_UPDATE = 0.5**2  # Gyr
EPS = 0.5 | units.kpc
HOST_N = 400000
SAT_N = 200000

RESULTS_DIR = "all_models/results/{}"
MODELS_DIR = "all_models/models/{}"


@dataclass
class Params:
    vx: float = 0
    vy: float = 0
    vz: float = 0
    h_xangle: float = 0
    h_yangle: float = 0
    s_xangle: float = 0
    s_yangle: float = 0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                attr: [getattr(self, attr)]
                for attr in (
                    "vx",
                    "vy",
                    "vz",
                    "h_xangle",
                    "h_yangle",
                    "s_xangle",
                    "s_yangle",
                )
            }
        )


MODELS = [
    # Params(vy=180),
    # Params(vy=180, h_yangle=np.pi/4),
    # Params(vy=180, h_yangle=np.pi/2),
    # Params(vy=180, s_xangle=np.pi/4),
    # Params(vy=180, s_xangle=np.pi/2),
    # Params(vx=-180 / 1.41, vy=180 / 1.41),
    # Params(vx=-180 / 1.41, vy=180 / 1.41, h_yangle=np.pi/4),
    # Params(vx=-180 / 1.41, vy=180 / 1.41, h_yangle=np.pi/2),
    # Params(vx=-180 / 1.41, vy=180 / 1.41, s_xangle=np.pi/4),
    # Params(vx=-180 / 1.41, vy=180 / 1.41, s_xangle=np.pi/2),
    Params(vx=-179, vy=19),
    Params(vx=-179, vy=19, h_yangle=np.pi / 4),
    Params(vx=-179, vy=19, h_yangle=np.pi / 2),
    Params(vx=-179, vy=19, s_xangle=np.pi / 4),
    Params(vx=-179, vy=19, s_xangle=np.pi / 2),
]


def _prepare_model(params: Params) -> Particles:
    host_particles = scriptslib.downsample(
        scriptslib.read_csv(MODELS_DIR.format("host.csv"), SPACE_UNIT, VEL_UNIT, MASS_UNIT),
        HOST_N,
    )
    sat_particles = scriptslib.downsample(
        scriptslib.read_csv(MODELS_DIR.format("sat.csv"), SPACE_UNIT, VEL_UNIT, MASS_UNIT),
        SAT_N,
    )

    if params.h_xangle != 0:
        scriptslib.rotate(host_particles, "x", params.h_xangle)
    if params.h_yangle != 0:
        scriptslib.rotate(host_particles, "y", params.h_yangle)
    if params.s_xangle != 0:
        scriptslib.rotate(host_particles, "x", params.s_xangle)
    if params.s_yangle != 0:
        scriptslib.rotate(host_particles, "y", params.s_yangle)

    sat_particles.position += [100, 0, 0] | units.kpc
    sat_particles.velocity += [params.vx, params.vy, params.vz] | units.kms

    particles = host_particles
    particles.add_particles(sat_particles)
    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    particles.id = range(len(particles))

    return particles


def _evolve_model(
    particles: Particles,
    until: float,
    *monitorings: Callable[[Particles, float], bool],
):
    time = 0

    while time < until:
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{time:.03f}")
        particles = physics.leapfrog(
            particles,
            EPS,
            DT | TIME_UNIT,
            SPACE_UNIT,
            VEL_UNIT,
            MASS_UNIT,
            TIME_UNIT,
        )

        need_break = False

        for monitoring in monitorings:
            need_break = need_break or monitoring(particles, time)

        if need_break:
            break

        time += DT


def model():
    """
    Evolve a lot of models with different
    1. initial velocity directions
    2. inclinations of the satellite galaxy
    3. inclinations of the host galaxy
    """

    ORBIT_QUEUE_LENGTH = 120
    BOUND_MASS_THRESHOLD = 0.08

    result = pd.DataFrame()

    for model in MODELS:
        particles_queue = deque()
        times_queue = deque()

        bound_masses = []
        times = []
        bound_ids = []

        particles = _prepare_model(model)
        initial_mass = particles[-SAT_N:].total_mass().value_in(MASS_UNIT)

        ##### Things to monitor during the evolution #####

        def compute_bound_mass(particles: Particles, time: float) -> bool:
            bound_subset = physics.bound_subset(particles[-SAT_N:], EPS, SPACE_UNIT, MASS_UNIT, VEL_UNIT)

            bound_masses.append(bound_subset.total_mass().value_in(MASS_UNIT))
            times.append(time)

            if bound_masses[-1] / initial_mass <= BOUND_MASS_THRESHOLD:
                bound_ids.extend(list(bound_subset.id))
                return True
            else:
                return False

        def save_last_snapshots(particles: Particles, time: float) -> bool:
            particles_queue.append(particles.copy_to_new_particles())
            times_queue.append(time)

            if len(particles_queue) > ORBIT_QUEUE_LENGTH:
                particles_queue.popleft()
                times_queue.popleft()

            return False

        _evolve_model(particles, MAX_TIME, save_last_snapshots, compute_bound_mass)

        if not bound_ids:
            print("Satellite was not disrupted")

        bound_particles_positions = np.zeros(shape=(ORBIT_QUEUE_LENGTH, 3))

        for i, snapshot in enumerate(particles_queue):
            bound_particles = snapshot[bound_ids]
            bound_particles_positions[i, :] = bound_particles.center_of_mass().value_in(SPACE_UNIT)

        print(bound_particles_positions)

        sma, e = ellipse_approx.fit_3d_ellipse(bound_particles_positions)
        print(sma, e)
        curr_result = model.to_dataframe()
        curr_result["sma"] = [sma]
        curr_result["e"] = [e]

        result = pd.concat([result, curr_result], ignore_index=True)
        result.to_csv(RESULTS_DIR.format("result.csv"))

        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        # plt.plot(bound_particles_positions[:, 0], bound_particles_positions[:, 1])
        # plt.show()
