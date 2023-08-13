from collections import deque
from datetime import datetime

import agama
import matplotlib.pyplot as plt
import numpy as np
from amuse.lab import Particles, ScalarQuantity, VectorQuantity, units

import scriptslib
from scriptslib import math
from scriptslib import particles as sparticles
from scriptslib import physics
from scriptslib import plot as splot
from scriptslib.log import log

MODELS_DIR = "system_generator/models/{}"


def generate_snapshot():
    MAX_TIME = 10.0 | units.Gyr
    DT = 0.5**7 | units.Gyr
    EPS = 0.1 | units.kpc
    HOST_N = 1000000
    SAT_N = 500000
    INCLINATION = np.deg2rad(0)
    LONG_ASC_NODE = np.deg2rad(45)
    output_times = deque(np.arange(0, 10, 0.5))
    position_unit_vector = [np.cos(INCLINATION), 0, np.sin(INCLINATION)] | units.kpc
    velocity_unit_vector = [
        -np.cos(INCLINATION) * np.cos(LONG_ASC_NODE),
        np.cos(INCLINATION) * np.sin(LONG_ASC_NODE),
        np.sin(INCLINATION),
    ] | units.kms

    host_particles = sparticles.pipe(
        scriptslib.read_csv(MODELS_DIR.format("host.csv")),
        sparticles.downsample(HOST_N),
        sparticles.set_attribute("system", "host"),
        sparticles.enumerate(),
        sparticles.set_attribute_by_condition(lambda id: id / HOST_N < 0.2, ["id"], "is_barion", True, False),
    )
    sat_particles = sparticles.pipe(
        scriptslib.read_csv(MODELS_DIR.format("sat.csv")),
        sparticles.downsample(SAT_N),
        sparticles.rotate("y", INCLINATION),
        sparticles.append_position(position_unit_vector * 100),
        sparticles.append_velocity(velocity_unit_vector * 180),
        sparticles.set_attribute("system", "sat"),
        sparticles.enumerate(),
        sparticles.set_attribute_by_condition(lambda id: id / SAT_N < 0.2, ["id"], "is_barion", True, False),
    )

    particles = sparticles.pipe(
        Particles(),
        sparticles.append_particles(host_particles),
        sparticles.append_particles(sat_particles),
        sparticles.move_to_origin(),
    )

    time = 0 | units.Gyr
    i = 0

    while time < MAX_TIME:
        log(datetime.now(), i, time.value_in(units.Gyr))
        particles = physics.leapfrog(particles, EPS, DT)

        if len(output_times) > 0 and time >= output_times[0] | units.Gyr:
            center = physics.median_iterative_center(particles, 8, 5 | units.kpc)
            particles.position -= center
            radii, densities = physics.get_density_distribution(
                particles,
                [0, 0, 0] | units.kpc,
                cutoff_radius=50 | units.kpc,
            )

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(20, 10)
            fig.suptitle(f"{time.value_in(units.Gyr):.03f} Gyr")
            ax1.set_xlabel("x, kpc")
            ax1.set_ylabel("y, kpc")
            splot.plot_hist(
                particles.x.value_in(units.kpc),
                particles.y.value_in(units.kpc),
                extent=[-100, 100, -100, 100],
                axes=ax1,
            )
            ax2.set_xlabel("r, kpc")
            ax2.set_ylabel("density, MSun/kpc^3")
            ax2.set_xlim(0, 50)
            ax2.grid(True)
            ax2.plot(
                radii.value_in(units.kpc),
                densities.value_in(units.MSun / units.kpc**3),
                label=f"{time.value_in(units.Gyr):.03f} Gyr",
            )
            ax2.set_yscale("log")
            plt.savefig(f"system_generator/output/{time.value_in(units.Gyr):.03f}.pdf")
            plt.cla()

            scriptslib.write_hdf5(particles, MODELS_DIR.format(f"{time.value_in(units.Gyr):.03f}.hdf5"))
            output_times.popleft()

        i += 1
        time += DT

    center = physics.median_iterative_center(particles, 8, 5 | units.kpc)
    particles.position -= center
    scriptslib.write_hdf5(particles, MODELS_DIR.format("particles.hdf5"))


if __name__ == "__main__":
    generate_snapshot()
