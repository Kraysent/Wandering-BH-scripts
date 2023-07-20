from dataclasses import dataclass

import agama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from amuse.lab import units

import scriptslib
from scriptslib import mnras, physics

RESULTS_DIR = "eccentricity_example/results/{}"

PERICENTRE = 5  # kpc
MAX_TIME = 4  # Gyr


def _get_df_in_potential(density_func, mass, ln_lambda, sigma):
    def df(pos, vel):
        """
        Dynamical friction formula in given potential.
        """

        r = sum(pos**2) ** 0.5
        v = sum(vel**2) ** 0.5
        rho = density_func(pos)
        X = v / (2**0.5 * sigma(r))

        return (
            -vel
            / v**3
            * (
                4
                * np.pi
                * agama.G**2
                * mass
                * rho
                * ln_lambda
                * (scipy.special.erf(X) - 2 / np.pi**0.5 * X * np.exp(-X * X))
            )
        )

    return df


def _get_ode_in_potential(potential, mass, ln_lambda):
    df_host = agama.DistributionFunction(type="quasispherical", potential=potential)
    grid_r = np.logspace(-1, 2, 16)
    grid_sig = (
        agama.GalaxyModel(potential, df_host).moments(
            np.column_stack((grid_r, grid_r * 0, grid_r * 0)),
            dens=False,
            vel=False,
            vel2=True,
        )[:, 0]
        ** 0.5
    )
    logspl = agama.CubicSpline(np.log(grid_r), np.log(grid_sig))
    sigma = lambda r: np.exp(logspl(np.log(r)))

    def ode(xv: np.ndarray, t: float):
        """
        ODE to be solved by scipy. xv is a state vector (x, y, z, vx, vy, vz).
        """

        df = _get_df_in_potential(potential.density, mass, ln_lambda, sigma)

        return np.concatenate((xv[3:6], potential.force(xv[:3], t=t) + df(xv[:3], xv[3:6])))

    return ode


def label_formatter(e, mass):
    exponent = len(str(int(mass))) - 1
    mantisse = mass / 10**exponent
    if mantisse != 1:
        return f"e: {e:.01f}, M: ${mantisse:.01f}\cdot 10^{exponent}\ M_{{\odot}}$"
    else:
        return f"e: {e:.01f}, M: $10^{exponent}\ M_{{\odot}}$"


@dataclass
class Parameters:
    eccentricity: float
    color: str
    linestyle: str
    mass: float


def model():
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    particles = scriptslib.read_csv("system_generator/models/particles.csv")
    potential = scriptslib.potential_from_particles(particles)

    params = [
        Parameters(0.0, "red", "dotted", 1e6),
        Parameters(0.4, "red", "dashed", 1e6),
        Parameters(0.7, "red", "solid", 1e6),
        Parameters(0.0, "green", "dotted", 1e7),
        Parameters(0.4, "green", "dashed", 1e7),
        Parameters(0.7, "green", "solid", 1e7),
        Parameters(0.0, "blue", "dotted", 1e8),
        Parameters(0.4, "blue", "dashed", 1e8),
        Parameters(0.7, "blue", "solid", 1e8),
    ]

    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    plt.tight_layout()
    fig1.set_size_inches(mnras.size_from_aspect(1))
    fig1.subplots_adjust(wspace=0, hspace=0)
    fig2.set_size_inches(mnras.size_from_aspect(1))
    fig2.subplots_adjust(wspace=0, hspace=0)

    for param in params:
        ode = _get_ode_in_potential(potential, param.mass, 5)
        # assuming the potential of the galaxy to pe spherically symmetrical.
        # This is not always the case (notably for galaxies with disk) but that should give quite good approximation.
        pericentre_velocity = np.sqrt(
            agama.G * potential.enclosedMass(PERICENTRE) * (1 + param.eccentricity) / PERICENTRE
        )
        print(f"{param.eccentricity}\t{pericentre_velocity}\t{PERICENTRE}")
        times = np.linspace(0, MAX_TIME, 2**10)

        ic = np.array([PERICENTRE, 0, 0, 0, pericentre_velocity, 0])
        traj = scipy.integrate.odeint(ode, ic, times)

        r = (traj[:, 0:3] ** 2).sum(axis=1) ** 0.5
        merger_index = np.argmax(r < 0.01)
        if merger_index == 0:
            merger_index = -1

        ax1.plot(
            traj[:, 0],
            traj[:, 1],
            color=param.color,
            label=label_formatter(param.eccentricity, param.mass),
            linewidth=1,
            linestyle=param.linestyle,
        )

        ax2.plot(
            times[:merger_index],
            r[:merger_index],
            color=param.color,
            label=label_formatter(param.eccentricity, param.mass),
            linewidth=1,
            linestyle=param.linestyle,
        )

    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-12, 12)
    ax2.set_xlim(0, MAX_TIME)
    ax2.set_ylim(0, 11)
    ax2.set_xlabel("Time, Gyr", fontsize=mnras.FONT_SIZE)
    ax2.set_ylabel("Distance, kpc", fontsize=mnras.FONT_SIZE)
    ax1.set_xlabel("x, kpc", fontsize=mnras.FONT_SIZE)
    ax1.set_ylabel("y, kpc", fontsize=mnras.FONT_SIZE)
    ax1.grid(True)
    ax2.grid(True)
    ax2.legend(fontsize=mnras.FONT_SIZE)
    ax1.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)
    ax2.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)
    fig1.savefig(RESULTS_DIR.format("eccentricity_orbits.pdf"), pad_inches=0, bbox_inches="tight")
    fig2.savefig(RESULTS_DIR.format("eccentricity_radii.pdf"), pad_inches=0, bbox_inches="tight")


if __name__ == "__main__":
    model()
