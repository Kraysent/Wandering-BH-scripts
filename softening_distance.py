from dataclasses import dataclass
from datetime import datetime
import agama
from amuse.lab import Particles, units
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scriptslib import physics, math
import scriptslib


def generate_spherical_system(n_points: int = 400000) -> Particles:
    potential_params = dict(type="spheroid", gamma=1, beta=3, scaleradius=4.2, densitynorm=1)
    vcirc1 = (-agama.Potential(potential_params).force(10, 0, 0)[0] * 10) ** 0.5
    potential_params["densitynorm"] = (200.0 / vcirc1) ** 2

    potential = agama.Potential(potential_params)
    df_host = agama.DistributionFunction(type="quasispherical", potential=potential)

    p_xv, p_m = agama.GalaxyModel(potential, df_host).sample(n_points)

    particles = Particles(size=n_points)
    particles.position = p_xv[:, :3] | units.kpc
    particles.velocity = p_xv[:, 3:] | units.kms
    particles.mass = p_m | units.MSun

    return particles


def read_system_from_file() -> Particles:
    return scriptslib.downsample(scriptslib.read_csv("models_velocity_vector/models/host.csv"), 400000)


@dataclass
class Parameters:
    eps_pc: float

    def __str__(self):
        return f"{self.eps_pc:.02f}"


params = [
    # Parameters(1),
    # Parameters(10),
    # Parameters(100),
    # Parameters(1000),
    # Parameters(5000),
    Parameters(10000),
]

if __name__ == "__main__":
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    DT = 0.5**7 | units.Gyr
    MAX_TIME = 10 | units.Gyr

    for param in params:
        time = 0 | units.Gyr
        i = 0
        particles = read_system_from_file()

        while time < MAX_TIME:
            print(f"{i}\t{datetime.now().strftime('%H:%M:%S')}\t{time.value_in(units.Gyr):.03f}")
            particles = physics.leapfrog(particles, param.eps_pc | units.pc, DT)

            time += DT

            if i % 200 == 0:
                radii = np.linspace(0.1, 15, 100)
                datapoints = np.zeros(shape=(len(radii), 3))
                datapoints[:, 0] = radii
                potential = agama.Potential(
                    type="multipole",
                    particles=(particles.position.value_in(units.kpc), particles.mass.value_in(units.MSun)),
                    lmax=0,
                )
                densities = potential.density(datapoints)

                plt.plot(radii, densities, label=f"{time.value_in(units.Gyr):.03f} Gyr")

            i += 1

        plt.legend()
        plt.title(f"Softening distance: {param.eps_pc} pc")
        plt.xlim(0, 15)
        plt.xlabel("r, kpc")
        plt.ylabel("density, MSun/kpc^3")
        plt.grid(True)
        plt.savefig(f"{str(param)}.pdf")
        plt.cla()
