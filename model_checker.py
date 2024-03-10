from random import randint

import agama
import numpy as np
from amuse.lab import units
from matplotlib import pyplot as plt

import scriptslib
from scriptslib import particles as sparticles
from scriptslib import physics


def get_circular_density(
    potential: agama.Potential, radii_span: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros(shape=(len(radii_span), 3))
    points[:, 0] = radii_span * np.cos(angle)
    points[:, 1] = radii_span * np.sin(angle)

    densities = potential.density(points)

    return radii_span, densities


def process():
    agama.setUnits(length=1, velocity=1, mass=1)
    particles = sparticles.pipe(
        scriptslib.read_hdf5("system_generator/models/particles.hdf5"),
        sparticles.downsample(100000),
        lambda particles: sparticles.append_position(physics.median_iterative_center(particles, 8, 5 | units.kpc))(particles),
        sparticles.align_angular_momentum(np.array([0, 0, 1])),
        log=True,
    )
    actual_potential = scriptslib.potential_from_particles(particles, lmax=15)

    for angle in np.linspace(0, 90, 2):
        radii_span = np.linspace(0.1, 25, 150)
        radii_dens, densities_actual = get_circular_density(actual_potential, radii_span, np.deg2rad(angle))

        plt.plot(radii_dens, densities_actual, label=f"${angle}$ deg.")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    process()
