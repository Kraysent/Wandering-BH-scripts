from configparser import RawConfigParser

import agama
import numpy as np


# write out the circular velocity curve for the entire model and per component
def writeRotationCurve(filename, potentials, names):
    radii = np.logspace(-3.0, 2.0, 101)
    xyz = np.column_stack((radii, radii * 0, radii * 0))
    vcomp2 = np.column_stack([-potential.force(xyz)[:, 0] * radii for potential in potentials])
    vtot2 = np.sum(vcomp2, axis=1)
    np.savetxt(
        filename,
        np.column_stack((radii, vtot2**0.5, vcomp2**0.5)),
        fmt="%.6g",
        header="radius\tVcTotal\t" + "\t".join(names),
    )


# print some diagnostic information after each iteration
def printoutInfo(model, iteration):
    densDisk = model.components[0].getDensity()
    densBulge = model.components[1].getDensity()
    densHalo = model.components[2].getDensity()
    pt0 = (2.0, 0, 0)
    pt1 = (2.0, 0, 0.25)
    pt2 = (0.0, 0, 2.0)
    print(
        "Disk  total mass=%g, rho(R=2,z=0)=%g, rho(R=2,z=0.25)=%g"
        % (densDisk.totalMass(), densDisk.density(pt0), densDisk.density(pt1))
    )
    print("Bulge total mass=%g, rho(R=0.5,z=0)=%g" % (densBulge.totalMass(), densBulge.density(0.4, 0, 0)))
    print(
        "Halo  total mass=%g, rho(R=2,z=0)=%g, rho(R=0,z=2)=%g"
        % (densHalo.totalMass(), densHalo.density(pt0), densHalo.density(pt2))
    )
    # report only the potential of stars+halo, excluding the potential of the central BH (0th component)
    pot0 = model.potential.potential(0, 0, 0)  # - model.potential[0].potential(0,0,0)
    print("Potential at origin=-(%g)^2, total mass=%g" % ((-pot0) ** 0.5, model.potential.totalMass()))
    densDisk.export(f"system_generator/output/misc/dens_disk_{iteration}")
    densBulge.export(f"system_generator/output/misc/dens_bulge_{iteration}")
    densHalo.export(f"system_generator/output/misc/dens_halo_{iteration}")
    model.potential.export(f"system_generator/output/misc/potential_{iteration}")
    # separate the contributions of bulge and halo, which are normally combined
    # into the Multipole potential of all spheroidal components
    writeRotationCurve(
        f"system_generator/output/misc/rotcurve_{iteration}",
        (
            # model.potential[0], # potential of the BH
            model.potential[1],  # potential of the disk
            agama.Potential(type="Multipole", lmax=6, density=densBulge),  # -"- bulge
            agama.Potential(type="Multipole", lmax=6, density=densHalo),
        ),
        ("BH", "Disk", "Bulge", "Halo"),
    )


def process():
    ini = RawConfigParser()
    ini.read("model_comparison/data/scm4.ini")
    potential_halo_params = dict(ini.items("Potential halo"))
    potential_bulge_params = dict(ini.items("Potential bulge"))
    potential_disk_params = dict(ini.items("Potential disk"))

    df_disk = dict(ini.items("DF disk"))
    scm_halo = dict(ini.items("SelfConsistentModel halo"))
    scm_bulge = dict(ini.items("SelfConsistentModel bulge"))
    scm_disk = dict(ini.items("SelfConsistentModel disk"))
    scm = dict(ini.items("SelfConsistentModel"))

    model = agama.SelfConsistentModel(**scm)

    density_disk = agama.Density(**potential_disk_params)
    density_bulge = agama.Density(**potential_bulge_params)
    density_halo = agama.Density(**potential_halo_params)

    model.components.append(agama.Component(density=density_disk, disklike=True))
    model.components.append(agama.Component(density=density_bulge, disklike=False))
    model.components.append(agama.Component(density=density_halo, disklike=False))

    model.iterate()
    printoutInfo(model, "init")

    # construct the DF of the disk component, using the initial (non-spherical) potential
    dfDisk = agama.DistributionFunction(potential=model.potential, **df_disk)
    # initialize the DFs of spheroidal components using the Eddington inversion formula
    # for their respective density profiles in the initial potential
    dfBulge = agama.DistributionFunction(type="QuasiSpherical", potential=model.potential, density=density_bulge)
    dfHalo = agama.DistributionFunction(type="QuasiSpherical", potential=model.potential, density=density_halo)

    print(
        "\033[1;33m**** STARTING ITERATIVE MODELLING ****\033[0m\nMasses (computed from DF): "
        "Mdisk=%g, Mbulge=%g, Mhalo=%g" % (dfDisk.totalMass(), dfBulge.totalMass(), dfHalo.totalMass())
    )

    # replace the initially static SCM components with the DF-based ones
    model.components[0] = agama.Component(df=dfDisk, disklike=True, **scm_disk)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **scm_bulge)
    model.components[2] = agama.Component(df=dfHalo, disklike=False, **scm_halo)

    # do a few more iterations to obtain the self-consistent density profile for both disks
    for iteration in range(1, 5):
        print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
        model.iterate()
        printoutInfo(model, "iter%d" % iteration)

    # export model to an N-body snapshot
    print("\033[1;33mCreating an N-body representation of the model\033[0m")

    # now create genuinely self-consistent models of both components,
    # by drawing positions and velocities from the DF in the given (self-consistent) potential
    print("Sampling disk DF")
    # agama.writeSnapshot("model_disk_final", \
    diskp, diskm = agama.GalaxyModel(potential=model.potential, df=dfDisk, af=model.af).sample(180000)
    print("Sampling bulge DF")
    bulgp, bulgm = agama.GalaxyModel(potential=model.potential, df=dfBulge, af=model.af).sample(20000)
    diskp = np.vstack((bulgp, diskp))
    diskm = np.hstack((bulgm, diskm))
    print("Sampling halo DF")
    Nhalo = 2000000
    halop, halom = agama.GalaxyModel(potential=model.potential, df=dfHalo, af=model.af).sample(Nhalo)

    # mass refinement for halo particles based on energy
    Eh = model.potential.potential(halop[:, 0:3]) + 0.5 * np.sum(halop[:, 3:6] ** 2, axis=1)
    order = np.argsort(Eh)
    halop = halop[order]
    # input range => output range;  mscale  count
    # 0    - 0.20    0    - 0.20    1       50%
    # 0.20 - 0.60    0.20 - 0.36    2.5     40%
    # 0.60 - 1.00    0.36 - 0.40    10      10%
    ind1 = Nhalo // 5
    ind2 = Nhalo * 3 // 5
    ind3 = Nhalo
    cnt1 = (ind2 - ind1) * 2 // 5
    cnt2 = (ind3 - ind2) // 10
    print(ind1, ind2, ind3, cnt1, cnt2)
    print(
        "Refinement at rc=%.3g (m*=2.5) and rc=%.3g (m*=10)" % tuple(model.potential.Rcirc(E=Eh[order][(ind1, ind2),]))
    )
    halop = np.vstack(
        (
            halop[:ind1],
            halop[ind1:ind2][np.random.choice(ind2 - ind1, cnt1)],
            halop[ind2:ind3][np.random.choice(ind3 - ind2, cnt2)],
        )
    )
    halom = np.hstack((np.ones(ind1), np.ones(cnt1) * 2.5, np.ones(cnt2) * 10)) * halom[0]
    agama.writeSnapshot(
        "system_generator/models/host_galaxy.nemo", (np.vstack((diskp, halop)), np.hstack((diskm, halom))), "n"
    )


if __name__ == "__main__":
    process()
