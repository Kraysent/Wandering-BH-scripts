from dataclasses import dataclass
from typing import Callable

import agama
import numpy as np
import scipy

import scriptslib
from scriptslib import particles as sparticles
from scriptslib import scheduling
from scriptslib.log import log as slog
from scriptslib.scheduling import SchedulerMode

RESULTS_DIR = "bh_sinking_times/results/{}"
MODELS_DIR = "bh_sinking_times/models/{}"

THRESHOLDS = [2, 5, 10, 20]  # pc
SMA_MAX = 30
MAX_TIME = 13.7  # Gyr
RESOLUTION = 30


@dataclass
class ParameterSet:
    bh_mass: float
    model_angular_momentum_direction: np.ndarray
    prefix: str = ""

    def __str__(self) -> str:
        return f"{self.bh_mass:.2E}, {self.prefix}"


parameters = [
    ParameterSet(2e6, np.array([0, 0, 1]), "in_plane/"),
    ParameterSet(1e7, np.array([0, 0, 1]), "in_plane/"),
    ParameterSet(1e8, np.array([0, 0, 1]), "in_plane/"),
    ParameterSet(2e6, np.array([0, np.sin(np.deg2rad(30)), 1 * np.cos(np.deg2rad(30))]), "rotated/"),
    ParameterSet(1e7, np.array([0, np.sin(np.deg2rad(30)), 1 * np.cos(np.deg2rad(30))]), "rotated/"),
    ParameterSet(1e8, np.array([0, np.sin(np.deg2rad(30)), 1 * np.cos(np.deg2rad(30))]), "rotated/"),
]


def get_df_in_potential(density_func, mass, ln_lambda, sigma):
    def df(pos, vel):
        """
        Dynamical friction formula in given potential.
        """

        r = sum(pos**2) ** 0.5
        v = sum(vel**2) ** 0.5
        rho = density_func(pos)
        X = v / (2**0.5 * sigma(r))

        df_formula = -vel / v**3 * (4 * np.pi * agama.G**2 * mass * rho * ln_lambda)
        df_formula *= scipy.special.erf(X) - 2 / np.pi**0.5 * X * np.exp(-X * X)
        return df_formula

    return df


def get_ode_in_potential(potential, mass, ln_lambda):
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
    df = get_df_in_potential(potential.density, mass, ln_lambda, sigma)

    def ode(xv: np.ndarray, t: float):
        """
        ODE to be solved by scipy. xv is a state vector (x, y, z, vx, vy, vz).
        """

        return np.concatenate((xv[3:6], potential.force(xv[:3], t=t) + df(xv[:3], xv[3:6])))

    return ode


def apocentre_velocity(a: float, e: float, potential: agama.Potential):
    apocentre = a * (1 + e)
    m = potential.enclosedMass(apocentre)
    return np.sqrt(agama.G * m / apocentre) * (1 - e)


def integrate(x0: np.ndarray, v0: np.ndarray, ode: Callable, times: np.ndarray) -> np.ndarray:
    ic = np.concatenate((x0, v0))
    return scipy.integrate.odeint(ode, ic, times)


def compute(params: ParameterSet):
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    potential = scriptslib.potential_from_particles(
        sparticles.pipe(
            scriptslib.read_hdf5(MODELS_DIR.format("particles.hdf5")),
            sparticles.downsample(1000000),
            sparticles.align_angular_momentum(params.model_angular_momentum_direction),
        ),
        lmax=15,
        symmetry="axisymmetric",
    )

    ecc_span = np.linspace(0.01, 0.99, RESOLUTION)
    sma_span = np.linspace(1, SMA_MAX, RESOLUTION)
    eccs, smas = np.meshgrid(ecc_span, sma_span, indexing="ij")
    sinking_times = {threshold: np.zeros(shape=eccs.shape) for threshold in THRESHOLDS}

    ode = get_ode_in_potential(potential, params.bh_mass, 5)
    times = np.linspace(0, MAX_TIME, 2**10)

    for iy, ix in np.ndindex(smas.shape):
        a, e = smas[iy, ix], eccs[iy, ix]
        velocity = apocentre_velocity(a, e, potential)

        trajectory = integrate([a, 0, 0], [0, velocity, 0], ode, times)
        rs = (trajectory[:, 0:3] ** 2).sum(axis=1) ** 0.5

        log_str = [eccs[iy, ix], smas[iy, ix]]

        for threshold in THRESHOLDS:
            index = np.argmax(rs < 0.001 * threshold)
            sinking_times[threshold][iy, ix] = MAX_TIME if index == 0 else times[index]
            log_str += [sinking_times[threshold][iy, ix]]

        slog(*log_str)

    for threshold in THRESHOLDS:
        np.savetxt(
            RESULTS_DIR.format(f"{params.prefix}bound_time_{params.bh_mass:.2E}_{threshold}.csv"),
            sinking_times[threshold],
            delimiter=",",
        )


if __name__ == "__main__":
    scheduler = scheduling.LinearScheduler(compute, parameters)
    scheduler.run(SchedulerMode.Multiprocessed)
