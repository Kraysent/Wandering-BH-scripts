import os
from dataclasses import dataclass
from typing import Callable

import agama
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy

from scriptslib import ellipse_approx, scheduling


@dataclass
class Params:
    mass: float
    size: float

    def __str__(self) -> str:
        return ""


# parameters for the simulation
T_END = 5.0  # total simulation time
T_UPD = 2**-8  # interval for plotting and updating the satellite mass for the restricted N-body simulation
# timestep of the full N-body sim (typically should be smaller than eps/v, where v is characteristic internal velocity)
TAU = 2**-8
EPS = 0.1  # softening length for the full N-body simulation
LN_LAMBDA = 3.0
SAT_NUMBER_OF_POINTS = 2000
SEMI_MAJOR_AXIS = 10.0
ECCENTRICITY = 0.1
APOCENTRE = (1 + ECCENTRICITY) * SEMI_MAJOR_AXIS
BOUND_MASS_THRESHOLD = 0.05

RESULTS_DIR = "nsc_models/results/{}"

# working units: 1 Msun, 1 kpc, 1 km/s


def dynamical_friction_acceleration(host_potential, pos, vel, mass, sigma: Callable[[float], float], ln_lambda: float):
    """
    compute the Chandrasekhar's dynamical friction acceleration for a point mass in the host galaxy
    """
    r = sum(pos**2) ** 0.5
    v = sum(vel**2) ** 0.5
    rho = host_potential.density(pos)
    X = v / (2**0.5 * sigma(r))
    return (
        -vel
        / v
        * (
            4
            * numpy.pi
            * agama.G**2
            * mass
            * rho
            * ln_lambda
            * (scipy.special.erf(X) - 2 / numpy.pi**0.5 * X * numpy.exp(-X * X))
            / v**2
        )
    )


def orbit_with_df(host_potential, ic, time, timestart, trajsize, mass, initmass, sigma, ln_lambda):
    """
    integrate the orbit of a massive particle in the host galaxy, accounting for dynamical friction
    """
    if mass == 0:
        return agama.orbit(ic=ic, potential=host_potential, time=time, timestart=timestart, trajsize=trajsize)
    times = numpy.linspace(timestart, timestart + time, trajsize)
    traj = scipy.integrate.odeint(
        lambda xv, t: numpy.hstack(
            (
                xv[3:6],
                host_potential.force(xv[:3], t=t)
                + dynamical_friction_acceleration(host_potential, xv[:3], xv[3:6], initmass, sigma, ln_lambda),
            )
        ),
        ic,
        times,
    )
    return times, traj


def setup_host_potential():
    pot_host_params = dict(type="spheroid", gamma=1, beta=3, scaleradius=4.2, densitynorm=1)
    vcirc1 = (-agama.Potential(pot_host_params).force(10, 0, 0)[0] * 10) ** 0.5
    pot_host_params["densitynorm"] = (200.0 / vcirc1) ** 2
    return agama.Potential(pot_host_params)


def prepare_velocity_dispersion_profile(pot_host):
    # prepare an interpolation table for the host velocity dispersion profile
    df_host = agama.DistributionFunction(type="quasispherical", potential=pot_host)
    grid_r = numpy.logspace(-1, 2, 16)  # grid from 0.1 to 100 kpc
    grid_sig = (
        agama.GalaxyModel(pot_host, df_host).moments(
            numpy.column_stack((grid_r, grid_r * 0, grid_r * 0)), dens=False, vel=False, vel2=True
        )[:, 0]
        ** 0.5
    )
    logspl = agama.CubicSpline(numpy.log(grid_r), numpy.log(grid_sig))  # log-scaled spline
    return lambda r: numpy.exp(logspl(numpy.log(r)))  # and the un-scaled interpolator


def initialize_satellite_potential(params: Params):
    # initial potential of the satellite (a single Dehnen component with a Gaussian cutoff)
    pot_sat = agama.Potential(
        type="spheroid",
        gamma=1,
        beta=4,
        scaleradius=params.size / 1000,
        outercutoffradius=8.0,
        mass=params.mass,
    )
    initmass = pot_sat.totalMass()

    # create a spherical isotropic DF for the satellite and sample it with particles
    df_sat = agama.DistributionFunction(type="quasispherical", potential=pot_sat)
    xv, mass = agama.GalaxyModel(pot_sat, df_sat).sample(SAT_NUMBER_OF_POINTS)
    return pot_sat, initmass, xv, mass


def process(sat_mass: float, sat_size: float):
    param = Params(sat_mass, sat_size)
    agama.setUnits(length=1, velocity=1, mass=1)

    pot_host = setup_host_potential()
    sigma = prepare_velocity_dispersion_profile(pot_host)
    pot_sat, initmass, xv, mass = initialize_satellite_potential(param)

    # place the satellite at the apocentre of a moderately eccentric orbit
    Vcirc = (-APOCENTRE * pot_host.force(APOCENTRE, 0, 0)[0]) ** 0.5
    V0 = (1 - ECCENTRICITY) * Vcirc

    # initial displacement
    r_center = numpy.array([APOCENTRE, 0, 0, 0, V0, 0])
    xv += r_center

    i = 0
    centres = []
    times = []

    r_mass = [initmass]
    r_traj = [r_center]
    r_xv = xv.copy()
    time = 0.0  # current simulation time
    times_t = [time]
    times_u = [time]
    # temp file name for storing the trajectory
    orbitfile = RESULTS_DIR.format(f"satellite_orbit_{param.mass:.02e}_{param.size:.02f}.txt")

    while time < T_END:
        times.append(time)
        i += 1
        # restricted N-body
        # first determine the trajectory of the satellite centre in the host potential
        # (assuming that it moves as a single massive particle)
        time_center, orbit_center = orbit_with_df(
            host_potential=pot_host,
            ic=r_center,
            time=T_UPD,
            timestart=time,
            trajsize=round(T_UPD / TAU) + 1,
            mass=r_mass[-1],
            initmass=initmass,
            sigma=sigma,
            ln_lambda=LN_LAMBDA,
        )
        times_u.append(time_center[-1])
        times_t.extend(time_center[1:])
        r_traj.extend(orbit_center[1:])
        r_center = orbit_center[-1]  # current position and velocity of satellite CoM
        # initialize the time-dependent total potential (host + moving sat) on this time interval
        centres.append(r_center)
        numpy.savetxt(orbitfile, numpy.column_stack((time_center, orbit_center)))
        try:
            pot_total = agama.Potential(pot_host, agama.Potential(potential=pot_sat, center=orbitfile))
        except RuntimeError:
            print(0, None)
            break

        centres_np = numpy.zeros(shape=(len(centres), 3))
        for j in range(len(centres)):
            centres_np[j] = centres[j][:3]

        # compute the trajectories of all particles moving in the combined potential of the host galaxy and the moving satellite
        r_xv = numpy.vstack(agama.orbit(ic=r_xv, potential=pot_total, time=T_UPD, timestart=time, trajsize=1)[:, 1])
        # update the potential of the satellite (using a spherical monopole approximation)
        pot_sat = agama.Potential(type="multipole", particles=(r_xv[:, :3] - r_center[:3], mass), lmax=0)
        # determine which particles remain bound to the satellite
        # ! why use only one iteration of bound mass determination?
        is_bound = (
            pot_sat.potential(r_xv[:, :3] - r_center[:3]) + 0.5 * numpy.sum((r_xv[:, 3:6] - r_center[3:6]) ** 2, axis=1)
            < 0
        )
        r_mass.append(numpy.sum(mass[is_bound]))

        if i % 50 == 0 or r_mass[-1] / initmass < BOUND_MASS_THRESHOLD:
            r_curr = ((r_xv[:, 0:3] - r_center[:3]) ** 2).sum(axis=1)
            dst_list = np.sort(r_curr[is_bound])
            size = dst_list[len(dst_list) // 2]
            a, e, _ = ellipse_approx.fit_2d_ellipse(centres_np[-50:, 0:2])
            print(a, e, size)

        time += T_UPD
        print(f"{i}  {time:5.3f}  {r_mass[-1] / initmass:.4g}")

        if r_mass[-1] / initmass < BOUND_MASS_THRESHOLD or time == T_END:
            os.remove(orbitfile)
            if time == T_END:
                print(f"T = {T_END}")

            fig, ax = plt.subplots()

            ax.grid(True)
            ax.set_xlabel("Time, Gyr")
            ax.set_ylabel("Mass, MSun")
            ax.plot(times_u, r_mass)
            fig.savefig(RESULTS_DIR.format(f"evolutions/mass_evolution_nsc_{param.mass:.02e}_{param.size:.02f}.pdf"))
            plt.close(fig)

            return r_mass[-1]


if __name__ == "__main__":
    scheduler = scheduling.MapScheduler(
        process,
        np.linspace(1e5, 5e7, 30),
        np.linspace(10.0, 50.0, 30),
    )

    results = scheduler.run(
        mode=scheduling.SchedulerMode.Multiprocessed
    )

    np.savetxt(RESULTS_DIR.format("results.csv"), results)
