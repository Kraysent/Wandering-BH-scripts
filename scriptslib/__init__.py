import numpy as np
import pandas as pd
from amuse.lab import Particles
from amuse.units import core

from scipy.spatial.transform import Rotation


def read_csv(
    path: str,
    space_unit: core.named_unit,
    vel_unit: core.named_unit,
    mass_unit: core.named_unit,
) -> Particles:
    df = pd.read_csv(path, delimiter=" ")

    particles = Particles(len(df))
    particles.x = np.array(df.x) | space_unit
    particles.y = np.array(df.y) | space_unit
    particles.z = np.array(df.z) | space_unit
    particles.vx = np.array(df.vx) | vel_unit
    particles.vy = np.array(df.vy) | vel_unit
    particles.vz = np.array(df.vz) | vel_unit
    particles.mass = np.array(df.m) | mass_unit

    return particles


def downsample(particles: Particles, to: int) -> Particles:
    coeff = len(particles) / to
    subset_indices = np.random.choice(len(particles), to, replace=False)
    new_particles = particles[subset_indices]
    new_particles.mass = new_particles.mass * coeff

    return new_particles


def rotate(model: Particles, axis: str, angle: float) -> Particles:
    if axis == "x":
        vec = np.array([1, 0, 0])
    elif axis == "y":
        vec = np.array([0, 1, 0])
    elif axis == "z":
        vec = np.array([0, 0, 1])
    else:
        raise ValueError("Unknown axis specified in rotation parameters.")

    rot_matrix = Rotation.from_rotvec(angle * vec).as_matrix()

    cm = model.center_of_mass()
    cm_vel = model.center_of_mass_velocity()
    model.position -= cm
    model.velocity -= cm_vel

    model.position = np.dot(model.position, rot_matrix.T)
    model.velocity = np.dot(model.velocity, rot_matrix.T)

    model.position += cm
    model.velocity += cm_vel

    return model
