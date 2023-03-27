import agama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from scriptslib import plot as splot

RESULTS_DIR = "bh_orbits/results/{}"
MODELS_DIR = "bh_orbits/models/{}"


def get_df_in_potential(potential, mass, ln_lambda, sigma):
    def df(pos, vel):
        """
        Dynamical friction formula in given potential.
        """

        r = sum(pos**2) ** 0.5
        v = sum(vel**2) ** 0.5
        rho = potential.density(pos)
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
    def ode(xv: np.ndarray, t: float):
        """
        ODE to be solved by scipy. xv is a state vector (x, y, z, vx, vy, vz).
        """

        df = get_df_in_potential(potential, mass, ln_lambda, lambda r: 200)

        return np.concatenate((xv[3:6], potential.force(xv[:3], t=t) + df(xv[:3], xv[3:6])))

    return ode


def compute():
    particles = pd.read_csv(RESULTS_DIR.format("particles.csv"))
    pos = particles[["x", "y", "z"]].to_numpy()
    mass = particles["mass"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    img = splot.plot_hist(red_x=pos[:, 0], red_y=pos[:, 1], extent=[-150, 150, -150, 150], axes=ax1)
    # res = ax1.scatter(pos[:, 0], pos[:, 1], c=mass, s=0.01)
    fig.colorbar(img)
    ax1.set_xlim(-150, 150)
    ax1.set_ylim(-150, 150)

    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    potential = agama.Potential(type="multipole", particles=(pos, mass), lmax=0)
    ode = get_ode_in_potential(potential, 1e8, 5)
    ic = np.array([5, 0, 0, 0, 180, 0])

    times = np.linspace(0, 13.7, 2**10)
    traj = scipy.integrate.odeint(ode, ic, times)
    rs = (traj**2).sum(axis=1) ** 2
    ax2.plot(times, rs)

    ax1.scatter(traj[:, 0], traj[:, 1], c=times, marker=".", cmap="Greys")
    plt.show()
