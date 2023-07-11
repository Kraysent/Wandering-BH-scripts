import functools
from typing import Callable
from amuse.lab import Particles, VectorQuantity
from scipy.spatial.transform import Rotation
import numpy as np


ParticlesFunc = Callable[[Particles], Particles]


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
