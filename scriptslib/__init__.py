import numpy as np
import pandas as pd
from amuse.lab import Particles
from amuse.units import core


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
