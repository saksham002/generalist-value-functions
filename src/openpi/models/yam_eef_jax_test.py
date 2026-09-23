"""Parity tests pinning the jnp YAM FK / delta ops to their numpy+scipy originals."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.yam_eef_jax as yam_eef_jax
import openpi.training.yam_eef as yam_eef
import openpi.transforms as _transforms

# The jnp path runs in float32 through a six-matrix chain; the numpy reference is float64.
TOLERANCE = 1e-4


def _random_joint_actions(shape: tuple[int, ...], seed: int = 86) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size = (*shape, yam_eef_jax.ACTION_DIM)).astype(np.float32)


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 5, 7)])
def test_joint_to_eef_matches_numpy(shape):
    actions = _random_joint_actions(shape)
    origins, axes = yam_eef_jax.chain_constants()

    actual = np.asarray(yam_eef_jax.joint_to_eef(jnp.asarray(actions), origins, axes))
    expected = yam_eef.joint_actions_to_eef(actions)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol = TOLERANCE)


def test_joint_to_eef_under_jit():
    actions = _random_joint_actions((3, 4))
    origins, axes = yam_eef_jax.chain_constants()

    jitted = jax.jit(lambda a: yam_eef_jax.joint_to_eef(a, origins, axes))
    np.testing.assert_allclose(
        np.asarray(jitted(jnp.asarray(actions))),
        yam_eef.joint_actions_to_eef(actions),
        atol = TOLERANCE,
    )


def test_euler_matrix_round_trip():
    rng = np.random.default_rng(86)
    # Stay clear of the pitch = +-pi/2 gimbal lock, where the euler decomposition is not unique.
    euler = rng.uniform(-1.2, 1.2, size = (16, 3)).astype(np.float32)

    matrix = yam_eef_jax.euler_xyz_to_matrix(jnp.asarray(euler))
    np.testing.assert_allclose(
        np.asarray(yam_eef_jax.matrix_to_euler_xyz(matrix)), euler, atol = TOLERANCE
    )


def test_euler_conversion_matches_scipy():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(86)
    euler = rng.uniform(-1.2, 1.2, size = (16, 3)).astype(np.float32)

    actual = np.asarray(yam_eef_jax.euler_xyz_to_matrix(jnp.asarray(euler)))
    expected = Rotation.from_euler("xyz", euler).as_matrix()
    np.testing.assert_allclose(actual, expected, atol = TOLERANCE)


@pytest.mark.parametrize("rpy_index_start", [None, (3, 10)])
def test_apply_delta_matches_transform(rpy_index_start):
    rng = np.random.default_rng(86)
    mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))
    state = rng.uniform(-1.0, 1.0, size = (2, 14)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size = (2, 5, 14)).astype(np.float32)

    actual = np.asarray(
        yam_eef_jax.apply_delta(jnp.asarray(actions), jnp.asarray(state), mask, rpy_index_start)
    )
    expected = _transforms.DeltaActions(mask = mask, rpy_index_start = rpy_index_start)(
        {"state": state, "actions": actions}
    )["actions"]

    np.testing.assert_allclose(actual, expected, atol = TOLERANCE)


@pytest.mark.parametrize("rpy_index_start", [None, (3, 10)])
def test_apply_absolute_matches_transform(rpy_index_start):
    rng = np.random.default_rng(86)
    mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))
    state = rng.uniform(-1.0, 1.0, size = (2, 14)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size = (2, 5, 14)).astype(np.float32)

    actual = np.asarray(
        yam_eef_jax.apply_absolute(jnp.asarray(actions), jnp.asarray(state), mask, rpy_index_start)
    )
    expected = _transforms.AbsoluteActions(mask = mask, rpy_index_start = rpy_index_start)(
        {"state": state, "actions": actions}
    )["actions"]

    np.testing.assert_allclose(actual, expected, atol = TOLERANCE)
