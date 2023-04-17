import os
import sys
import agama
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scriptslib import plot as splot
from scriptslib import mnras
from matplotlib import figure
import json


RESULTS_DIR = "bh_orbits/results/{}"
MODELS_DIR = "bh_orbits/models/{}"

BH_MASS = 5e8 # MSun
THRESHOLDS = [2, 5, 10, 20]  # pc
MAX_TIME = 13.7  # Gyr
RESOLUTION = 20

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
            np.column_stack((grid_r, grid_r * 0, grid_r * 0)), dens=False, vel=False, vel2=True
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


def _get_potential_from_particles(particles: pd.DataFrame) -> agama.Potential:
    pos = particles[["x", "y", "z"]].to_numpy()
    mass = particles["mass"].to_numpy()
    return agama.Potential(type="multipole", particles=(pos, mass), lmax=0)

def _get_apocentre_velocity(a: float, e: float, potential: agama.Potential):
    apocentre = a * (1 + e)
    m = potential.enclosedMass(apocentre)
    return np.sqrt(agama.G * m / apocentre) * (1 - e)


def compute(debug: bool = False, additional_results: str | None = None):
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr
    particles = pd.read_csv(RESULTS_DIR.format("particles.csv"))
    potential = _get_potential_from_particles(particles)

    ecc_span = np.linspace(0.01, 0.99, RESOLUTION)
    sma_span = np.linspace(1, 16, RESOLUTION)
    eccs, smas = np.meshgrid(ecc_span, sma_span, indexing="ij")
    bound_times = {threshold: np.zeros(shape=eccs.shape) for threshold in THRESHOLDS}
    velocities = np.zeros(shape=eccs.shape)

    ode = _get_ode_in_potential(potential, BH_MASS, 5)
    times = np.linspace(0, MAX_TIME, 2**10)

    for iy, ix in np.ndindex(smas.shape):
        a, e = smas[iy, ix], eccs[iy, ix]
        velocities[iy, ix] = _get_apocentre_velocity(a, e, potential)

        ic = np.array([a, 0, 0, 0, velocities[iy, ix], 0])
        traj = scipy.integrate.odeint(ode, ic, times)

        rs = (traj[:, 0:3] ** 2).sum(axis=1) ** 0.5

        log = f"({eccs[iy, ix]:.02f}, {smas[iy, ix]:.02f}):"

        for threshold in THRESHOLDS:
            index = np.argmax(rs < 0.001 * threshold)

            bound_times[threshold][iy, ix] = MAX_TIME if index == 0 else times[index]

            log += f"\t{bound_times[threshold][iy, ix]:.02f}"

        print(log)

        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure.figaspect(1 / 2))
            ax1.set_xlim(-150, 150)
            ax1.set_ylim(-150, 150)
            ax2.grid(True)
            ax2.set_ylim(0, 16)

            splot.plot_hist(red_x=particles["x"], red_y=particles["y"], extent=[-150, 150, -150, 150], axes=ax1)
            ax1.scatter(traj[:, 0], traj[:, 1], c=times, marker=".", cmap="Greys")
            ax2.plot(times, rs)

            fig.savefig(RESULTS_DIR.format(f"debug/{a:.02f}-{e:.02f}.pdf"))
            plt.close(fig)

    for threshold in [2, 5, 10, 20]:
        np.savetxt(RESULTS_DIR.format(f"bound_time_{BH_MASS:.2E}_{threshold}.csv"), bound_times[threshold], delimiter=",")

    _plot(bound_times, additional_results)

# no loading functionality yet

def _prepare_axes(ax):
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 16)
    ax.set_xlabel("Eccentricity", fontsize=mnras.FONT_SIZE)
    ax.set_ylabel("Semi-major axis, kpc", fontsize=mnras.FONT_SIZE)
    ax.set_title(f"M = {BH_MASS:.0E} MSun", fontsize=mnras.FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=mnras.FONT_SIZE)

def _plot(bound_times: dict[float, np.ndarray], additional_results: str | None):
    for threshold in [2, 5, 10, 20]:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(mnras.size_from_aspect(0.6))
        _prepare_axes(ax)
        pic = ax.imshow(
            bound_times[threshold].T[::-1, :],
            extent=[0.01, 0.99, 0.01, 16],
            interpolation="nearest",
            aspect="auto",
            cmap="gray",
        )
        cbar = plt.colorbar(pic)
        cbar.set_label("Time till BH's complete sinking, Gyr", fontsize=mnras.FONT_SIZE)
        cbar.ax.tick_params(labelsize=mnras.FONT_SIZE)

        if additional_results is not None:
            with open(MODELS_DIR.format(additional_results), "r") as j:
                results = json.loads(j.read())

            print(results)

            ecc, sma, colors, markers, fills = (
                results["eccentricities"],
                results["majsemiaxes"],
                results["colors"],
                results["markers"],
                results["fills"],
            )

            for i in range(len(ecc)):
                ax.scatter(
                    ecc[i],
                    sma[i],
                    c=colors[i],
                    edgecolor="black",
                    marker=MarkerStyle(markers[i], fillstyle=fills[i]),
                )

        plt.tight_layout()
        fig.savefig(RESULTS_DIR.format(f"result_{BH_MASS:.2E}_{threshold}.pdf"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
