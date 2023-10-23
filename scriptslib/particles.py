from typing import Callable

import agama
import numpy as np
from amuse.lab import Particles, VectorQuantity, units
from scipy.spatial.transform import Rotation

ParticlesFunc = Callable[[Particles], Particles]


def generate_spherical_system(n_points: int = 400000) -> Particles:
    potential_params = dict(
        type="spheroid",
        gamma=1,
        beta=3,
        scaleradius=4.2,
        densitynorm=1,
        outerCutoffRadius=100,
    )
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


def pipe(particles: Particles, *functions: ParticlesFunc):
    for func in functions:
        particles = func(particles)

    return particles


def append_particles(new_set: Particles) -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        if len(particles) == 0:
            return new_set

        particles.add_particles(new_set)
        return particles

    return wrapper


def downsample(to: int) -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        coeff = len(particles) / to
        subset_indices = np.random.choice(len(particles), to, replace=False)
        new_particles = particles[subset_indices]
        new_particles.mass = new_particles.mass * coeff

        return new_particles

    return wrapper


def rotate(axis: str, angle: float) -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        if axis == "x":
            vec = np.array([1, 0, 0])
        elif axis == "y":
            vec = np.array([0, 1, 0])
        elif axis == "z":
            vec = np.array([0, 0, 1])
        else:
            raise ValueError("Unknown axis specified in rotation parameters.")

        if angle == 0:
            return particles

        rot_matrix = Rotation.from_rotvec(angle * vec).as_matrix()

        cm = particles.center_of_mass()
        cm_vel = particles.center_of_mass_velocity()
        particles.position -= cm
        particles.velocity -= cm_vel

        particles.position = np.dot(particles.position, rot_matrix.T)
        particles.velocity = np.dot(particles.velocity, rot_matrix.T)

        particles.position += cm
        particles.velocity += cm_vel

        return particles

    return wrapper


def append_position(to: VectorQuantity) -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        particles.position += to

        return particles

    return wrapper


def append_velocity(to: VectorQuantity) -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        particles.velocity += to

        return particles

    return wrapper


def move_to_origin() -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        particles.position -= particles.center_of_mass()
        particles.velocity -= particles.center_of_mass_velocity()

        return particles

    return wrapper


def enumerate() -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        particles.id = np.arange(0, len(particles))

        return particles

    return wrapper


def set_attribute(name: str, value) -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        setattr(particles, name, value)

        return particles

    return wrapper


def set_attribute_by_condition(
    condition: Callable,
    condition_attributes: list[str],
    attribute: str,
    value_if_true,
    value_if_false,
):
    def wrapper(particles: Particles) -> Particles:
        setattr(particles, attribute, value_if_false)
        p = particles.select(condition, condition_attributes)
        setattr(p, attribute, value_if_true)

        return particles

    return wrapper


def select(condition: Callable, attributes: list[str]) -> ParticlesFunc:
    def wrapper(particles: Particles) -> Particles:
        p = particles.select(condition, attributes)

        return p

    return wrapper
