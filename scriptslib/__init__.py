from collections import namedtuple
from dataclasses import dataclass

import agama
import numpy as np
import pandas as pd
from amuse.lab import Particles
from amuse.lab import units as aunits
from amuse.units import core
from scipy.spatial.transform import Rotation


@dataclass
class Units:
    space: core.named_unit = None
    vel: core.named_unit = None
    mass: core.named_unit = None
    time: core.named_unit = None


default_units = Units(space=aunits.kpc, vel=aunits.kms, mass=232500 * aunits.MSun, time=aunits.Gyr)


def potential_from_particles(particles: Particles) -> agama.Potential:
    pos = particles.position.value_in(aunits.kpc)
    mass = particles.mass.value_in(aunits.MSun)
    return agama.Potential(type="multipole", particles=(pos, mass), lmax=0)


def read_csv(
    path: str,
    space_unit: core.named_unit | None = None,
    vel_unit: core.named_unit | None = None,
    mass_unit: core.named_unit | None = None,
    units: Units | None = default_units,
) -> Particles:
    df = pd.read_csv(path, sep=" ")

    if space_unit or vel_unit or mass_unit:
        units = Units(space=space_unit, vel=vel_unit, mass=mass_unit)

    particles = Particles(len(df))
    particles.x = np.array(df.x) | units.space
    particles.y = np.array(df.y) | units.space
    particles.z = np.array(df.z) | units.space
    particles.vx = np.array(df.vx) | units.vel
    particles.vy = np.array(df.vy) | units.vel
    particles.vz = np.array(df.vz) | units.vel
    particles.mass = np.array(df.m) | units.mass

    return particles


def write_csv(particles: Particles, path: str, units: Units = default_units):
    output_table = pd.DataFrame()
    output_table["x"] = particles.x.value_in(units.space)
    output_table["y"] = particles.y.value_in(units.space)
    output_table["z"] = particles.z.value_in(units.space)
    output_table["vx"] = particles.vx.value_in(units.vel)
    output_table["vy"] = particles.vy.value_in(units.vel)
    output_table["vz"] = particles.vz.value_in(units.vel)
    output_table["m"] = particles.mass.value_in(units.mass)
    output_table.to_csv(path, sep=" ")


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
