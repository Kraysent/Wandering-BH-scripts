import os
import sys
import agama
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scriptslib import plot as splot
from amuse.lab import units, constants
from matplotlib import figure
import json


RESULTS_DIR = "bh_orbits/results/{}"
MODELS_DIR = "bh_orbits/models/{}"


def get_df_in_potential(density_func, mass, ln_lambda, sigma):
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


def get_ode_in_potential(potential, mass, ln_lambda):
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

        df = get_df_in_potential(potential.density, mass, ln_lambda, sigma)

        return np.concatenate((xv[3:6], potential.force(xv[:3], t=t) + df(xv[:3], xv[3:6])))

    return ode


def compute(debug: bool = False, additional_results: str | None = None):
    particles = pd.read_csv(RESULTS_DIR.format("particles.csv"))
    pos = particles[["x", "y", "z"]].to_numpy()
    mass = particles["mass"].to_numpy()

    ecc_span = np.linspace(0.01, 0.99, 6)
    sma_span = np.linspace(1, 16, 6)
    eccs, smas = np.meshgrid(ecc_span, sma_span, indexing="ij")
    bound_times = {
        20: np.zeros(shape=eccs.shape),
        10: np.zeros(shape=eccs.shape),
        5: np.zeros(shape=eccs.shape),
        2: np.zeros(shape=eccs.shape),
    }
    velocities = np.zeros(shape=eccs.shape)

    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    potential = agama.Potential(type="multipole", particles=(pos, mass), lmax=0)
    ode = get_ode_in_potential(potential, 5e8, 5)
    times = np.linspace(0, 13.7, 2**10)

    def cum_mass(r: float):
        return potential.enclosedMass(r)

    for iy, ix in np.ndindex(smas.shape):
        a, e = smas[iy, ix], eccs[iy, ix]
        apocentre = a * (1 + e)
        m = cum_mass(apocentre)

        vel = np.sqrt(agama.G * m / apocentre) * (1 - e)
        velocities[iy, ix] = vel

        ic = np.array([a, 0, 0, 0, vel, 0])

        traj = scipy.integrate.odeint(ode, ic, times)
        rs = (traj[:, 0:3] ** 2).sum(axis=1) ** 0.5

        log = f"{eccs[iy, ix]:.02f},\t{smas[iy, ix]:.02f}:"

        for threshold in [2, 5, 10, 20]:
            index = np.argmax(rs < 0.001 * threshold)

            bound_times[threshold][iy, ix] = 13.7 if index == 0 else times[index]

            log += f"\t{bound_times[threshold][iy, ix]:.02f}"

        print(log)

        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure.figaspect(1 / 2))
            ax1.set_xlim(-150, 150)
            ax1.set_ylim(-150, 150)
            ax2.grid(True)
            ax2.set_ylim(0, 16)
            splot.plot_hist(red_x=pos[:, 0], red_y=pos[:, 1], extent=[-150, 150, -150, 150], axes=ax1)
            ax1.scatter(traj[:, 0], traj[:, 1], c=times, marker=".", cmap="Greys")
            ax2.plot(times, rs)

            fig.savefig(RESULTS_DIR.format(f"debug/{a:.02f}-{e:.02f}.pdf"))
            plt.close(fig)

    for threshold in [2, 5, 10, 20]:
        np.savetxt(RESULTS_DIR.format(f"bound_time_{threshold}.csv"), bound_times[threshold], delimiter=",")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figure.figaspect(1) * 3)

    for ax, size in ((ax1, 2), (ax2, 5), (ax3, 10), (ax4, 20)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 16)
        pic = ax.imshow(
            bound_times[size].T[::-1, :], extent=[0.01, 0.99, 0.01, 16], interpolation="nearest", aspect="auto", cmap="gray"
        )
        plt.colorbar(pic)

    if additional_results is not None:
        with open(MODELS_DIR.format(additional_results), "r") as j:
            results = json.loads(j.read())

        for ax in (ax1, ax2, ax3, ax4):
            ecc, sma, colors, markers, fills = (
                results["eccentricities"],
                results["majsemiaxes"],
                results["colors"],
                results["markers"],
                results["fills"],
            )

            for i, _ in enumerate(ecc):
                ax.scatter(
                    ecc[i],
                    sma[i],
                    c=colors[i],
                    edgecolor="black",
                    marker=MarkerStyle(markers[i], fillstyle=fills[i]),
                )

    fig.savefig(RESULTS_DIR.format("result.pdf"))
