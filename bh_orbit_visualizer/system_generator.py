from datetime import datetime
from amuse.lab import units
import numpy as np
import scriptslib
from scriptslib import physics

MAX_TIME = 10.0 | units.Gyr
DT = 0.5**7 | units.Gyr
EPS = 0.5 | units.kpc
HOST_N = 400000
SAT_N = 200000
INCLINATION = np.deg2rad(30)
LONG_ASC_NODE = np.deg2rad(45)

INPUT_DIR = "bh_orbit_visualizer/input/{}"


def generate_snapshot():
    host_particles = scriptslib.downsample(scriptslib.read_csv(INPUT_DIR.format("host.csv")), HOST_N)
    sat_particles = scriptslib.downsample(scriptslib.read_csv(INPUT_DIR.format("sat.csv")), SAT_N)
    scriptslib.rotate(sat_particles, "y", INCLINATION)

    sat_particles.position += [
        100 * np.cos(INCLINATION),
        0,
        100 * np.sin(INCLINATION),
    ] | units.kpc
    sat_particles.velocity += [
        -180 * np.cos(INCLINATION) * np.cos(LONG_ASC_NODE),
        180 * np.cos(INCLINATION) * np.sin(LONG_ASC_NODE),
        180 * np.sin(INCLINATION),
    ] | units.kms

    particles = host_particles
    particles.add_particles(sat_particles)

    time = 0 | units.Gyr

    while time < MAX_TIME:
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{time.value_in(units.Gyr):.03f}")
        particles = physics.leapfrog(particles, EPS, DT)

        time += DT

    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    scriptslib.write_csv(particles, INPUT_DIR.format("particles.csv"))
