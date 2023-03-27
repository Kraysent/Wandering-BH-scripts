from datetime import datetime

import pandas as pd
from amuse.lab import units

import scriptslib
from scriptslib import physics

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

MAX_TIME = 6.0  # Gyr
DT = 0.5**6  # Gyr
DT_UPDATE = 0.5**2  # Gyr
EPS = 0.5 | units.kpc
PLOT_ITERATION = int(DT_UPDATE / DT)
HOST_N = 400000
SAT_N = 200000

RESULTS_DIR = "bh_orbits/results/{}"
MODELS_DIR = "bh_orbits/models/{}"


def generate_snapshot():
    host_particles = scriptslib.downsample(
        scriptslib.read_csv(MODELS_DIR.format("host.csv"), SPACE_UNIT, VEL_UNIT, MASS_UNIT),
        HOST_N,
    )
    sat_particles = scriptslib.downsample(
        scriptslib.read_csv(MODELS_DIR.format("sat.csv"), SPACE_UNIT, VEL_UNIT, MASS_UNIT),
        SAT_N,
    )

    sat_particles.position += [100, 0, 0] | units.kpc
    sat_particles.velocity += [-180 / 1.41, 180 / 1.41, 0] | units.kms

    particles = host_particles
    particles.add_particles(sat_particles)

    time = 0

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

        time += DT

    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    output_table = pd.DataFrame()

    for c in ["x", "y", "z"]:
        output_table[c] = getattr(particles, c).value_in(SPACE_UNIT)

    for c in ["vx", "vy", "vz"]:
        output_table[c] = getattr(particles, c).value_in(VEL_UNIT)

    output_table["mass"] = particles.mass.value_in(units.MSun)
    output_table.to_csv(RESULTS_DIR.format("particles.csv"))
