from concurrent import futures
from dataclasses import dataclass
from datetime import datetime

import agama
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from amuse.lab import units
from matplotlib import pyplot as plt
import multiprocessing

import scriptslib
from scriptslib import physics, particles as sparticles
import system_generator

RESULT_DIR = "softening_distance/output/{}"
DT = 0.5**7 | units.Gyr
MAX_TIME = 10 | units.Gyr


@dataclass
class Parameters:
    eps_pc: float
    number_of_particles: int
    name: str = ""

    def __str__(self):
        if self.name != "":
            return f"{self.name}-{self.eps_pc:.02f}-{self.number_of_particles:.00f}"
        else:
            return f"{self.eps_pc:.02f}-{self.number_of_particles:.00f}"


params = [
    Parameters(1000, 500000, "host"),
    Parameters(900, 500000, "host"),
    Parameters(800, 500000, "host"),
    Parameters(700, 500000, "host"),
    Parameters(600, 500000, "host"),
    Parameters(400, 500000, "host"),
    Parameters(300, 500000, "host"),
    Parameters(200, 500000, "host"),
    Parameters(100, 500000, "host"),
]


def process(param: Parameters):
    time = 0 | units.Gyr
    i = 0
    particles = sparticles.pipe(
        scriptslib.read_csv("system_generator/models/host.csv"),
        sparticles.downsample(param.number_of_particles),
    )

    radii_history = []
    densities_history = []
    times = []

    while time < MAX_TIME:
        particles = physics.leapfrog(particles, param.eps_pc | units.pc, DT)

        if i % 200 == 0:
            radii, densities = system_generator.get_density_distribution(
                particles,
                physics.median_iterative_center(particles, 8, 5 | units.kpc),
                cutoff_radius=15 | units.kpc,
                resolution=50,
            )
            radii_history.append(radii.value_in(units.kpc))
            densities_history.append(densities.value_in(units.MSun / units.kpc**3))
            times.append(time)

        i += 1
        time += DT

        yield time

    fig, ax1 = plt.subplots()
    ax2: Axes = inset_axes(ax1, width="60%", height="70%", loc="upper right", borderpad=1.2)

    for radii, densities, time in zip(radii_history, densities_history, times):
        ax2.plot(radii, densities, label=f"{time.value_in(units.Gyr):.03f} Gyr")
        ax1.plot(radii, densities_history[0] - densities, label=f"{time.value_in(units.Gyr):.03f} Gyr")

    ax2.legend()
    fig.suptitle(f"eps: {param.eps_pc} pc | N: {param.number_of_particles}")
    ax1.set_xlim(0, 15)
    ax1.set_xlabel("$r$, kpc")
    ax1.set_ylabel("density residuals, $M_{\odot}$ kpc$^{-3}$")
    ax1.grid(True)
    ax2.set_xlim(0, 15)
    ax2.set_xlabel("$r$, kpc")
    ax2.set_ylabel("density, $M_{\odot}$ kpc$^{-3}$")
    ax2.grid(True)
    fig.savefig(RESULT_DIR.format(f"{str(param)}.pdf"))

    yield None


def process_with_log(param: Parameters):
    for i, time in enumerate(process(param)):
        if time is None:
            break

        print(f"{param.name}\t{i}\t{datetime.now().strftime('%H:%M:%S')}\t{time.value_in(units.Gyr):.03f}")


if __name__ == "__main__":
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    with futures.ProcessPoolExecutor(max_workers=9) as executor:
        executor.map(process_with_log, params)
