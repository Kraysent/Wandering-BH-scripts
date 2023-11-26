import agama
from amuse.lab import Particles, units

from scriptslib import physics


def generate_system(scale_velocity: float = 200.0, number_of_particles: int = 5000):
    potential_params = dict(
        type="spheroid",
        gamma=1,
        beta=3,
        scaleradius=4.2,
        densitynorm=1,
        outerCutoffRadius=100,
    )
    vcirc1 = (-agama.Potential(potential_params).force(10, 0, 0)[0] * 10) ** 0.5
    potential_params["densitynorm"] = (scale_velocity / vcirc1) ** 2
    potential = agama.Potential(potential_params)
    df_host = agama.DistributionFunction(type="quasispherical", potential=potential)
    p_xv, p_m = agama.GalaxyModel(potential, df_host).sample(number_of_particles)

    particles = Particles(size=number_of_particles)
    particles.position = p_xv[:, :3] | units.kpc
    particles.velocity = p_xv[:, 3:] | units.kms
    particles.mass = p_m | units.MSun

    return particles


def test_median_iterative_center_units():
    "mostly a sanity check"
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr
    particles = generate_system()
    iterative_center = physics.median_iterative_center(particles, 1, 10 | units.kpc)

    iterative_center.value_in(units.kpc)


def test_median_iterative_center_spherical_system_origin():
    "mostly a sanity check"
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr
    particles = generate_system()
    cm_center = particles.center_of_mass()
    iterative_center = physics.median_iterative_center(particles, 1, 10 | units.kpc)

    assert (cm_center - iterative_center).length().value_in(units.kpc) < 1


def test_median_iterative_center_spherical_system_offset():
    "mostly a sanity check"
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr
    particles = generate_system()
    particles.position += [0, 50, 50] | units.kpc
    cm_center = particles.center_of_mass()
    iterative_center = physics.median_iterative_center(particles, 1, 10 | units.kpc)

    assert (cm_center - iterative_center).length().value_in(units.kpc) < 1


def test_median_iterative_center_host_sat():
    "mostly a sanity check"
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    host_particles = generate_system(number_of_particles=10000)
    host_center = host_particles.center_of_mass()
    sat_particles = generate_system(scale_velocity=100, number_of_particles=3000)
    sat_particles.position += [0, 50, 50] | units.kpc
    particles = host_particles.copy()
    particles.add_particles(sat_particles)

    iterative_center = physics.median_iterative_center(particles, 4, 5 | units.kpc)

    assert (host_center - iterative_center).length().value_in(units.kpc) < 1


def test_median_iterative_center_host_sat_different_resolution():
    "mostly a sanity check"
    agama.setUnits(mass=1, length=1, velocity=1)  # 1 MSun, 1 kpc, 1 kms => time = 0.98 Gyr

    host_particles = generate_system(number_of_particles=10000)
    host_center = host_particles.center_of_mass()
    sat_particles = generate_system(scale_velocity=100, number_of_particles=10000)
    print(f"{host_particles.total_mass().value_in(units.MSun):.02e}")
    print(f"{sat_particles.total_mass().value_in(units.MSun):.02e}")
    sat_particles.position += [0, 50, 50] | units.kpc
    particles = host_particles.copy()
    particles.add_particles(sat_particles)

    iterative_center = physics.median_iterative_center(particles, 4, 3 | units.kpc)

    assert (host_center - iterative_center).length().value_in(units.kpc) < 1
