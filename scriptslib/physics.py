import pyfalcon
from amuse.lab import Particles, ScalarQuantity, units
from amuse.units import core


def leapfrog(
    particles: Particles,
    eps: ScalarQuantity,
    dt: ScalarQuantity,
    space_unit: core.named_unit,
    vel_unit: core.named_unit,
    mass_unit: core.named_unit,
    time_unit: core.named_unit,
) -> Particles:
    """
    Performs one step of integration using pyfalcon.
    """
    dt = dt.value_in(time_unit)
    acceleration, _ = pyfalcon.gravity(
        particles.position.value_in(space_unit),
        particles.mass.value_in(mass_unit),
        eps.value_in(space_unit),
    )

    particles.velocity += acceleration * dt / 2 | vel_unit
    particles.position += particles.velocity * (dt | time_unit)

    acceleration, _ = pyfalcon.gravity(
        particles.position.value_in(space_unit),
        particles.mass.value_in(mass_unit),
        eps.value_in(space_unit),
    )

    particles.velocity += acceleration * dt / 2 | vel_unit

    return particles


vector_length = lambda v, unit, axis=0: (v.value_in(unit) ** 2).sum(axis=axis) ** 0.5


def distance(host: Particles, satellite: Particles, unit: core.named_unit) -> float:
    return vector_length(host.center_of_mass() - satellite.center_of_mass(), unit)


def bound_subset(
    galaxy: Particles,
    eps: ScalarQuantity,
    space_unit: core.named_unit,
    mass_unit: core.named_unit,
    vel_unit: core.named_unit,
    number_of_iterations: int = 15,
    change_threshold: float = 0.01,
) -> Particles:
    change = 1.0
    curr_len, prev_len = len(galaxy), 0
    i = 0

    while change >= change_threshold:
        _, potentials = pyfalcon.gravity(
            galaxy.position.value_in(space_unit),
            galaxy.mass.value_in(mass_unit),
            eps.value_in(space_unit),
        )
        velocities = vector_length(
            galaxy.velocity - galaxy.center_of_mass_velocity(), vel_unit, axis=1
        )
        full_specific_energies = potentials + velocities**2 / 2

        galaxy = galaxy[full_specific_energies < 0]

        prev_len = curr_len
        curr_len = len(galaxy)

        if curr_len == 0 or prev_len == 0:
            break

        change = (prev_len - curr_len) / prev_len
        i += 1

        if i >= number_of_iterations:
            break

    return galaxy


def bound_mass(
    galaxy: Particles,
    eps: ScalarQuantity,
    space_unit: core.named_unit,
    mass_unit: core.named_unit,
    vel_unit: core.named_unit,
    number_of_iterations: int = 15,
    change_threshold: float = 0.01,
) -> float:
    return bound_subset(galaxy, eps, space_unit, mass_unit, vel_unit, number_of_iterations, change_threshold).total_mass().value_in(mass_unit)
