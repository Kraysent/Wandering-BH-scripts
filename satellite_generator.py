#!/usr/bin/env python

import agama, numpy
from amuse.lab import Particles, units

import scriptslib

MODELS_DIR = "system_generator/models/{}"

N_DISK = 100000
N_HALO = int(400000 / 0.4)
scale = 2.5


def process():
    stars_component = agama.Potential(type="sersic", mass=5e8 / 232500 * scale ** (1/3), scaleradius=2.6 * scale ** (1/3), sersicindex=1)
    dm_component = agama.Potential(
        type="spheroid",
        mass=2e11 / 232500 * scale,
        scaleradius=12 * scale ** (1/3),
        outercutoffradius=60 * scale ** (1/3),
        gamma=1,
        beta=3,
        cutoffstrength=4,
    )

    pot = agama.Potential(stars_component, dm_component)
    stars_df = agama.DistributionFunction(type="QuasiSpherical", density=stars_component, potential=pot)
    dm_df = agama.DistributionFunction(type="QuasiSpherical", density=dm_component, potential=pot)
    print(f"Mstar={stars_df.totalMass():.4g}={stars_component.totalMass():.4g}")
    print(f"Mhalo={dm_df.totalMass():.4g}={dm_component.totalMass():.4g}")
    stars_model = agama.GalaxyModel(pot, stars_df)
    dm_model = agama.GalaxyModel(pot, dm_df)

    diskp, diskm = stars_model.sample(N_DISK)
    halop, halom = dm_model.sample(N_HALO)

    # mass refinement for halo particles based on energy
    Ed = pot.potential(diskp[:, 0:3]) + 0.5 * numpy.sum(diskp[:, 3:6] ** 2, axis=1)
    diskp = diskp[numpy.argsort(Ed)]
    Eh = pot.potential(halop[:, 0:3]) + 0.5 * numpy.sum(halop[:, 3:6] ** 2, axis=1)
    order = numpy.argsort(Eh)
    halop = halop[order]

    # input range => output range;  mscale  count
    # 0    - 0.20    0    - 0.20    1       50%
    # 0.20 - 0.60    0.20 - 0.36    2.5     40%
    # 0.60 - 1.00    0.36 - 0.40    10      10%
    ind1 = N_HALO // 5
    ind2 = N_HALO * 3 // 5
    ind3 = N_HALO
    cnt1 = (ind2 - ind1) * 2 // 5
    cnt2 = (ind3 - ind2) // 10
    print(ind1, ind2, ind3, cnt1, cnt2)
    print(
        "Refinement at rc={:.3g} (m*=2.5) and rc={:.3g} (m*=10)".format(
            *tuple(pot.Rcirc(E=Eh[order][(ind1, ind2),])),
        )
    )
    halop = numpy.vstack(
        (
            halop[:ind1],
            halop[ind1:ind2][numpy.random.choice(ind2 - ind1, cnt1, replace=False)],
            halop[ind2:ind3][numpy.random.choice(ind3 - ind2, cnt2, replace=False)],
        )
    )

    halom = (
        numpy.hstack(
            (numpy.ones(ind1), numpy.ones(cnt1) * 2.5, numpy.ones(cnt2) * 10),
        )
        * halom[0]
    )

    xv = numpy.vstack((diskp, halop))
    m = numpy.hstack((diskm, halom))
    particles = Particles(xv.shape[0])
    particles.position = xv[:, :3] | units.kpc
    particles.velocity = xv[:, 3:] | units.kms
    particles.mass = m | 232500 * units.MSun

    scriptslib.write_hdf5(particles, MODELS_DIR.format("sat5.hdf5"))

if __name__ == "__main__":
    process()
