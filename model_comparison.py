import agama
from matplotlib import figure, pyplot as plt
import numpy as np

import scriptslib
from scriptslib import particles as sparticles

DATA_DIR = "./model_comparison/data/{}"


def get_circular_velocity(potential: agama.Potential, radii_span: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros(shape=(len(radii_span), 3))
    points[:, 0] = radii_span
    potentials = potential.potential(points)

    pot_diff = np.diff(potentials)
    radii_diff = np.diff(radii_span)
    v_circ = np.sqrt(radii_span[:-1] * pot_diff / radii_diff)

    return radii_span[:-1], v_circ


def get_circular_density(potential: agama.Potential, radii_span: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.zeros(shape=(len(radii_span), 3))
    points[:, 0] = radii_span
    densities = potential.density(points)

    # [1:] to make shape identical to get_circular_velocity function
    return radii_span[1:], densities[1:]


if __name__ == "__main__":
    agama.setUnits(length=1, velocity=1, mass=1)
    radii_span = np.linspace(0, 100, 150)
    expected_potential = agama.Potential(DATA_DIR.format("mcmillan17.ini"))
    radii, v_circ_expected = get_circular_velocity(expected_potential, radii_span)
    radii_dens, densities_expected = get_circular_density(expected_potential, radii_span)

    particles = sparticles.pipe(
        scriptslib.read_nemo("system_generator/models/host_galaxy.nemo"),
        sparticles.downsample(10000),
    )
    actual_potential = scriptslib.potential_from_particles(particles)

    radii, v_circ_actual = get_circular_velocity(actual_potential, radii_span)
    radii_dens, densities_actual = get_circular_density(actual_potential, radii_span)

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(figure.figaspect(1 / 2))
    axes[0, 0].plot(radii, v_circ_expected, label="McMillan17")
    axes[0, 0].plot(radii, v_circ_actual, label="SCM4")
    axes[0, 0].set_xlabel("$r$, kpc")
    axes[0, 0].set_ylabel("$v_{circ}$, km/s")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[1, 0].plot(radii, v_circ_actual - v_circ_expected, label="diff")
    axes[1, 0].grid(True)
    axes[1, 0].set_ylabel("$v_{McMillan} - v_{merger}$")
    axes[1, 0].legend()

    axes[0, 1].plot(radii_dens, densities_expected, label="McMillan17")
    axes[0, 1].plot(radii_dens, densities_actual, label="SCM4")
    axes[0, 1].set_xlabel("$r$, kpc")
    axes[0, 1].set_ylabel("$\\rho$")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 1].plot(radii_dens, densities_expected / densities_actual, label="diff")
    axes[1, 1].set_ylabel("$\\rho_{McMillan}/\\rho_{merger}$")
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    plt.show()