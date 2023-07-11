from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import agama
import scipy

RESULTS_DIR = "bh_orbits/results/{}"
MODELS_DIR = "bh_orbits/models/{}"

BH_MASS = 1e8  # MSun
PERICENTRE = 5  # kpc
MAX_TIME = 1.5  # Gyr


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


@dataclass
class Parameters:
    eccentricity: float
    color: str
    linestyle: str
    mass: float


def model():
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    pot_host_params = dict(type="spheroid", gamma=1, beta=3, scaleradius=4.2, densitynorm=1)
    vcirc1 = (-agama.Potential(pot_host_params).force(10, 0, 0)[0] * 10) ** 0.5
    pot_host_params["densitynorm"] = (200.0 / vcirc1) ** 2
    potential = agama.Potential(pot_host_params)

    params = [
        Parameters(0.0, "red", "solid", 1e7),
        Parameters(0.4, "green", "solid", 1e7),
        Parameters(0.9, "blue", "solid", 1e7),
        Parameters(0.0, "red", "dashed", 1e8),
        Parameters(0.4, "green", "dashed", 1e8),
        Parameters(0.7, "blue", "dashed", 1e8),
        Parameters(0.0, "red", "dotted", 5e8),
        Parameters(0.4, "green", "dotted", 5e8),
        Parameters(0.7, "blue", "dotted", 5e8),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2)

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

        error_filter = r < 15

        ax1.plot(
            traj[:, 0][error_filter],
            traj[:, 1][error_filter],
            color=param.color,
            label=f"e = {param.eccentricity:.02f}, M = {param.mass:.02e}",
            linewidth=1,
            linestyle=param.linestyle,
        )
        ax2.plot(
            times[error_filter],
            r[error_filter],
            color=param.color,
            label=f"e = {param.eccentricity:.02f}, M = {param.mass:.02e}",
            linewidth=1,
            linestyle=param.linestyle,
        )

    plt.gcf().set_size_inches(20, 10)
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-12, 12)
    ax2.set_xlim(0, MAX_TIME)
    ax2.set_ylim(0, 11)
    ax2.set_xlabel("Time, Gyr")
    ax2.set_ylabel("Distance, kpc")
    ax1.set_xlabel("x, kpc")
    ax1.set_ylabel("y, kpc")
    ax1.grid(True)
    ax2.grid(True)
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    model()
