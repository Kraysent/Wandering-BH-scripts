import agama
import matplotlib.pyplot as plt
from datetime import datetime
from amuse.lab import units, Particles, VectorQuantity, ScalarQuantity
import numpy as np
import scriptslib
from scriptslib import physics, plot as splot, particles as sparticles
from scriptslib import math
from scriptslib.log import log

MODELS_DIR = "system_generator/models/{}"


def get_density_distribution(
    particles: Particles,
    center: VectorQuantity = [0, 0, 0] | units.kpc,
    resolution=100,
    cutoff_radius: ScalarQuantity = 200 | units.kpc,
) -> tuple[VectorQuantity, VectorQuantity]:
    """
    General algorithm:
    - sort radii and masses together
    - put radii into a histogram and get indices of the bin that correspond to given radius
    - take masses with corresponding indices and sum all of the masses in a given bin of the histogram.
    - divide masses in these bin by the volume of the corresponding bin: (4/3 * pi * (r_1^3 - r_2^3))
    """
    radii = math.get_lengths(particles.position - center)
    masses = particles.mass
    radii, masses = math.sort_with(radii, masses)

    bins = np.linspace(0, cutoff_radius.value_in(units.kpc), resolution) | units.kpc
    # indicies always != 0 because r is never < 0.
    indicies = np.digitize(radii.value_in(units.kpc), bins.value_in(units.kpc))
    # hence, bincount will always have 0 at the index 0 and we can skip it.
    layer_masses = np.bincount(indicies, weights=masses.value_in(units.MSun))[1:-1] | units.MSun
    layer_volumes = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    return bins[1:], layer_masses / layer_volumes


def generate_snapshot():
    MAX_TIME = 10.0 | units.Gyr
    DT = 0.5**7 | units.Gyr
    EPS = 0.1 | units.kpc
    HOST_N = 1000000
    SAT_N = 500000
    INCLINATION = np.deg2rad(0)
    LONG_ASC_NODE = np.deg2rad(45)
    position_unit_vector = [np.cos(INCLINATION), 0, np.sin(INCLINATION)] | units.kpc
    velocity_unit_vector = [
        -np.cos(INCLINATION) * np.cos(LONG_ASC_NODE),
        np.cos(INCLINATION) * np.sin(LONG_ASC_NODE),
        np.sin(INCLINATION),
    ] | units.kms

    host_particles = sparticles.pipe(
        scriptslib.read_csv(MODELS_DIR.format("host.csv")),
        sparticles.downsample(HOST_N),
    )
    sat_particles = sparticles.pipe(
        scriptslib.read_csv(MODELS_DIR.format("sat.csv")),
        sparticles.downsample(SAT_N),
        sparticles.rotate("y", INCLINATION),
        sparticles.append_position(position_unit_vector * 100),
        sparticles.append_velocity(velocity_unit_vector * 180),
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
        time += DT

        if i % 100 == 0:
            radii, densities = get_density_distribution(
                particles,
                particles[:HOST_N].center_of_mass(),
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
            # ax2.set_yscale("log")
            plt.savefig(f"system_generator/output/{time.value_in(units.Gyr):.03f}.pdf")
            plt.cla()

        i += 1

    scriptslib.write_csv(particles, MODELS_DIR.format("particles.csv"))


if __name__ == "__main__":
    generate_snapshot()
