from amuse.lab import Particles, Particle, units
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scriptslib

## This command draws the position of the BH incorrectly for some reason. TODO: check this one more time.

SPACE_UNIT = units.kpc
VEL_UNIT = units.kms
MASS_UNIT = 232500 * units.MSun
TIME_UNIT = units.Gyr

N = 30000
MASS = 1e10  # MSun
BOX = (30, 10, 1)  # kpc
BH_INIT_POS = [0, -1, 0] | units.kpc

MAX_TIME = 1 | units.Gyr
EPS = 0.5 | units.kpc
DT = 0.5**9 | units.Gyr
DT_UPDATE = 0.5**8 | units.Gyr
PLOT_ITERATION = int(DT_UPDATE / DT)

RESULTS_DIR = "dynamical_friction_example/results/{}"


def _prepare_axis(ax):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-11, -9)
    ax.set_xlabel("x, kpc")
    ax.set_ylabel("y, kpc")


def model():
    fig, ax = plt.subplots()
    plt.tight_layout()

    std_dev = 1  # kms
    velocity_set = np.random.normal(0, std_dev, (N, 3)) | units.kms
    x_set = np.random.uniform(-BOX[0] / 2, BOX[0] / 2, N) | units.kpc
    y_set = np.random.uniform(-BOX[1] / 2, BOX[1] / 2, N) | units.kpc
    z_set = np.random.uniform(-BOX[2] / 2, BOX[2] / 2, N) | units.kpc

    background_particles = scriptslib.downsample(
        scriptslib.read_csv(
            "models_resolution/models/host.csv", SPACE_UNIT, VEL_UNIT, MASS_UNIT
        ),
        N,
    )

    bh_particle = Particle()
    bh_particle.position = BH_INIT_POS
    bh_particle.velocity = [50, 0, 0] | units.kms
    bh_particle.mass = 1e9 | units.MSun

    particles = background_particles
    particles.add_particle(bh_particle)

    time = 0 | units.Gyr
    i = 0

    while time < MAX_TIME:
        print(f"{datetime.now().strftime('%H:%M:%S')}\t{time.value_in(units.Myr):.03f}")
        particles = scriptslib.leapfrog(
            particles, EPS, DT, SPACE_UNIT, VEL_UNIT, MASS_UNIT, TIME_UNIT
        )

        if i % PLOT_ITERATION == 0:
            dsts = ((particles.position - BH_INIT_POS).value_in(units.kpc) ** 2).sum(
                axis=1
            ) ** 0.5
            dsts_filter = dsts < 3
            vis_particles = particles[dsts_filter]
            ax.clear()
            _prepare_axis(ax)
            ax.plot(
                vis_particles.x[:-1].value_in(units.kpc),
                vis_particles.y[:-1].value_in(units.kpc),
                marker=",",
                color="r",
                linestyle="None",
            )
            ax.plot(
                vis_particles.x[-1].value_in(units.kpc),
                vis_particles.x[-1].value_in(units.kpc),
                color="b",
                linestyle="None",
                marker="o",
            )
            ax.set_title(f"{time.value_in(units.Myr):.03f} Myr")

            fig.savefig(RESULTS_DIR.format(f"{time.value_in(units.Myr):.03f}.pdf"))

        i += 1
        time += DT
