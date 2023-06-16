import numpy as np
import pyfalcon
from amuse.lab import Particles, ScalarQuantity, units, VectorQuantity
from amuse.units import core
from scriptslib import Units, default_units


def leapfrog(
    particles: Particles,
    eps: ScalarQuantity,
    dt: ScalarQuantity,
    space_unit: core.named_unit | None = None,
    vel_unit: core.named_unit | None = None,
    mass_unit: core.named_unit | None = None,
    time_unit: core.named_unit | None = None,
    units: Units | None = default_units,
) -> Particles:
    """
    Performs one step of integration using pyfalcon.
    """
    if space_unit or vel_unit or mass_unit or time_unit:
        units = Units(space=space_unit, vel=vel_unit, mass=mass_unit, time=time_unit)

    dt = dt.value_in(units.time)
    acceleration, _ = pyfalcon.gravity(
        particles.position.value_in(units.space),
        particles.mass.value_in(units.mass),
        eps.value_in(units.space),
    )

    particles.velocity += acceleration * dt / 2 | units.vel
    particles.position += particles.velocity * (dt | units.time)

    acceleration, _ = pyfalcon.gravity(
        particles.position.value_in(units.space),
        particles.mass.value_in(units.mass),
        eps.value_in(units.space),
    )

    particles.velocity += acceleration * dt / 2 | units.vel

    return particles


vector_length = lambda v, unit, axis=0: (v.value_in(unit) ** 2).sum(axis=axis) ** 0.5


def distance(host: Particles, satellite: Particles, unit: core.named_unit) -> float:
    return vector_length(host.center_of_mass() - satellite.center_of_mass(), unit)


def get_potentials(
    particles: Particles,
    eps: ScalarQuantity,
    space_unit: core.named_unit,
    mass_unit: core.named_unit,
    time_unit: core.named_unit,
) -> VectorQuantity:
    pos = particles.position.value_in(space_unit)
    mass = particles.mass.value_in(mass_unit)
    eps = eps.value_in(space_unit)

    _, pot = pyfalcon.gravity(pos, mass, eps)

    return pot | space_unit**2 / time_unit**2


def potential_centre(
    particles: Particles,
    potentials: VectorQuantity | None = None,
    eps: ScalarQuantity | None = None,
    space_unit: core.named_unit | None = None,
    mass_unit: core.named_unit | None = None,
    time_unit: core.named_unit | None = None,
    top_fraction: float = 0.01,
) -> VectorQuantity:
    if potentials is None:
        potentials = get_potentials(particles, eps, space_unit, mass_unit, time_unit)

    perm = potentials.argsort()
    positions = particles.position[perm]
    positions = positions[: int(len(positions) * top_fraction)]
    masses = particles.mass[perm]
    masses = masses[: int(len(masses) * top_fraction)]

    return np.sum(positions * masses[:, np.newaxis], axis=0) / np.sum(masses)


def potential_centre_velocity(
    particles: Particles,
    potentials: VectorQuantity | None = None,
    eps: ScalarQuantity | None = None,
    space_unit: core.named_unit | None = None,
    mass_unit: core.named_unit | None = None,
    time_unit: core.named_unit | None = None,
    top_fraction: float = 0.01,
) -> VectorQuantity:
    if potentials is None:
        potentials = get_potentials(particles, eps, space_unit, mass_unit, time_unit)

    perm = potentials.argsort()
    velocities = particles.velocity[perm]
    velocities = velocities[: int(len(velocities) * top_fraction)]
    masses = particles.mass[perm]
    masses = masses[: int(len(masses) * top_fraction)]

    return np.sum(velocities * masses[:, np.newaxis], axis=0) / np.sum(masses)


def bound_subset(
    galaxy: Particles,
    eps: ScalarQuantity,
    space_unit: core.named_unit | None = None,
    mass_unit: core.named_unit | None = None,
    vel_unit: core.named_unit | None = None,
    units: Units | None = default_units,
    number_of_iterations: int = 15,
    change_threshold: float = 0.01,
) -> Particles:
    if space_unit or mass_unit or vel_unit:
        units = Units(space=space_unit, vel=vel_unit, mass=mass_unit)

    change = 1.0
    curr_len, prev_len = len(galaxy), 0
    i = 0

    while change >= change_threshold:
        _, potentials = pyfalcon.gravity(
            galaxy.position.value_in(units.space),
            galaxy.mass.value_in(units.mass),
            eps.value_in(units.space),
        )
        velocities = vector_length(galaxy.velocity - galaxy.center_of_mass_velocity(), units.vel, axis=1)
        full_specific_energies = potentials + velocities**2 / 2

        new_bound_subset = galaxy[full_specific_energies < 0]
        # if there are no particles in bound subset, return last iteration
        if len(new_bound_subset) == 0:
            return galaxy

        galaxy = new_bound_subset

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
    return (
        bound_subset(
            galaxy,
            eps,
            space_unit,
            mass_unit,
            vel_unit,
            number_of_iterations,
            change_threshold,
        )
        .total_mass()
        .value_in(mass_unit)
    )
