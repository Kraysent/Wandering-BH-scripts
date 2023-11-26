import numpy as np
import pandas
import pytest
from amuse.lab import Particles, units
from scriptslib.particles import align_angular_momentum

@pytest.mark.parametrize(
    "initial_position, initial_velocity, expected_angular_momentum",
    [
        # single particle
        ([np.array([1, 0, 0])], [np.array([0, 1, 0])], np.array([0, 1, 0])),
        # two symmetrical particles
        ([np.array([1, 0, 0]), np.array([-1, 0, 0])], [np.array([0, 1, 0]), np.array([0, -1, 0])], np.array([0, 1, 0])),
        # two asymmetrical particles
        ([np.array([1, 0, 0]), np.array([0, 1, 0])], [np.array([0, -1, 0]), np.array([1, 0, 0])], np.array([0, 1, 0])),
        # four random (but fixed) particles
        (
            [np.array([1, 0.5, 0]), np.array([0, 1, -0.1]), np.array([-5, 1, 0]), np.array([4, 3, -2])],
            [np.array([3, -2, 1]), np.array([1, 3, 2]), np.array([-3, -2.4, 2]), np.array([-1, 0.2, 2])],
            np.array([0, 1, 1]),
        ),
    ],
)
def test_align_angular_momentum_happy_path(initial_position, initial_velocity, expected_angular_momentum):
    # Arrange
    particles = Particles(len(initial_position))
    particles.position = initial_position | units.m
    particles.velocity = initial_velocity | units.m / units.s
    particles.mass = 1 | units.kg

    expected_angular_momentum = expected_angular_momentum / (expected_angular_momentum ** 2).sum() ** 0.5

    # Act
    aligned_particles = align_angular_momentum(expected_angular_momentum)(particles)
    actual_vector = aligned_particles.total_angular_momentum()

    # Assert
    assert np.allclose(
        actual_vector / actual_vector.length(), expected_angular_momentum
    ), "The particles' angular momentum should be aligned with the target vector."
