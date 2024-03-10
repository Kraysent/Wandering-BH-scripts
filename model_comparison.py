import agama
import numpy as np
from matplotlib import figure
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import scriptslib
from scriptslib import mnras
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
    radii_span = np.linspace(0, 60, 150)
    expected_potential = agama.Potential(DATA_DIR.format("mcmillan17.ini"))
    radii, v_circ_expected = get_circular_velocity(expected_potential, radii_span)
    radii_dens, densities_expected = get_circular_density(expected_potential, radii_span)

    particles = sparticles.pipe(
        scriptslib.read_hdf5("system_generator/models/merger.hdf5"),
        sparticles.downsample(1000000),
    )
    actual_potential = scriptslib.potential_from_particles(particles)

    radii, v_circ_actual = get_circular_velocity(actual_potential, radii_span)
    radii_dens, densities_actual = get_circular_density(actual_potential, radii_span)

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(figure.figaspect(1) * 2)
    axes[0].plot(radii, v_circ_expected, label="McMillan model")
    axes[0].plot(radii, v_circ_actual, label="Merger result")
    axes[0].set_ylabel("$v_{circ}$, km/s", fontsize=mnras.FONT_SIZE)
    axes[0].grid(True)
    axes[0].set_xlim(0, 60)
    axes[0].legend(loc="lower left", fontsize=mnras.FONT_SIZE)
    axes[0].tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    v_inset = inset_axes(axes[0], width="50%", height="60%", loc="lower right", borderpad=0)
    v_inset.plot(radii, v_circ_actual - v_circ_expected)
    v_inset.grid(True)
    v_inset.set_ylabel("$v_{McMillan} - v_{merger}$, km/s", fontsize=mnras.FONT_SIZE)
    v_inset.set_xticklabels([])
    v_inset.set_xlim(0, 60)
    v_inset.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    axes[1].plot(radii_dens, densities_expected, label="McMillan model")
    axes[1].plot(radii_dens, densities_actual, label="Merger result")
    axes[1].set_xlabel("$r$, kpc", fontsize=mnras.FONT_SIZE)
    axes[1].set_ylabel("$\\rho, M_{\\ocirc}/kpc$,", fontsize=mnras.FONT_SIZE)
    axes[1].set_yscale("log")
    axes[1].set_xlim(0, 60)
    axes[1].grid(True)
    axes[1].tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    rho_inset = inset_axes(axes[1], width="50%", height="60%", loc="upper right", borderpad=0)
    rho_inset.plot(radii_dens, densities_expected / densities_actual)
    rho_inset.set_ylabel("$\\rho_{McMillan}/\\rho_{merger}$", fontsize=mnras.FONT_SIZE)
    rho_inset.set_xlabel("$r$, kpc", fontsize=mnras.FONT_SIZE)
    rho_inset.set_xlim(0, 60)
    rho_inset.grid(True)
    rho_inset.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

    plt.subplots_adjust(hspace=0)
    plt.savefig("model_comparison/results/models_comparison.pdf", bbox_inches="tight")
    plt.show()
