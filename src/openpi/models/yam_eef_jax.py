"""JAX port of the YAM joint -> EEF forward kinematics.

`openpi.training.yam_eef` performs this conversion with mujoco + scipy, which cannot run
inside a JIT graph. The BestOfN eval path needs the conversion *between* sampling and
critic scoring, both of which live in one traced function, so the FK chain is re-expressed
in `jnp` here.

The chain constants (six body origins + hinge axes) are still read from the pinned
kinematics model via `yam_eef._load_fk`, so the flange (`link6`) convention and
`vendor/yam_vendor_kin.xml` remain the single source of truth shared with the RLDS
builder. Rotations are emitted as extrinsic-xyz euler to match
`scipy Rotation.as_euler("xyz")`, which is what `transforms.DeltaActions` /
`AbsoluteActions` and the cached counterfactual store use.
"""

import functools

import jax.numpy as jnp
import numpy as np

from openpi.shared import array_typing as at
from openpi.training import yam_eef

JOINTS_PER_ARM = yam_eef.JOINTS_PER_ARM
ACTION_DIM = yam_eef.ACTION_DIM


@functools.lru_cache(maxsize=2)
def chain_constants(yam_fk_dir: str = yam_eef.DEFAULT_YAM_FK_DIR) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load the FK chain as jnp constants: origins ``(6, 4, 4)`` and unit hinge axes ``(6, 3)``."""
    fk = yam_eef._load_fk(yam_fk_dir)  # noqa: SLF001
    return jnp.asarray(fk._origins, dtype = jnp.float32), jnp.asarray(fk._axes, dtype = jnp.float32)  # noqa: SLF001


def _rodrigues(
    axes: at.Float[at.Array, "j 3"],
    angles: at.Float[at.Array, "*b j"],
) -> at.Float[at.Array, "*b j 3 3"]:
    """Rotation matrices about each unit `axes[j]` by `angles[..., j]`."""
    x, y, z = axes[..., 0], axes[..., 1], axes[..., 2]
    zero = jnp.zeros_like(x)
    skew = jnp.stack(
        [
            jnp.stack([zero, -z, y], axis = -1),
            jnp.stack([z, zero, -x], axis = -1),
            jnp.stack([-y, x, zero], axis = -1),
        ],
        axis = -2,
    )
    sin = jnp.sin(angles)[..., None, None]
    cos = jnp.cos(angles)[..., None, None]
    eye = jnp.eye(3, dtype = axes.dtype)
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def fk(
    joint_angles: at.Float[at.Array, "*b 6"],
    origins: at.Float[at.Array, "6 4 4"],
    axes: at.Float[at.Array, "6 3"],
) -> tuple[at.Float[at.Array, "*b 3"], at.Float[at.Array, "*b 3 3"]]:
    """One arm's 6 joint angles (rad) -> flange position and rotation matrix in the base frame.

    Mirrors `yam_fk.YamFK.fk`, except the orientation is returned as a rotation matrix
    rather than a quaternion so downstream composition needs no round-trip.
    """
    joint_rotations = _rodrigues(axes, joint_angles)
    links = jnp.broadcast_to(origins, (*joint_angles.shape[:-1], *origins.shape))
    links = links.at[..., :3, :3].set(origins[..., :3, :3] @ joint_rotations)

    flange = links[..., 0, :, :]
    for i in range(1, JOINTS_PER_ARM):
        flange = flange @ links[..., i, :, :]
    return flange[..., :3, 3], flange[..., :3, :3]


def matrix_to_euler_xyz(matrix: at.Float[at.Array, "*b 3 3"]) -> at.Float[at.Array, "*b 3"]:
    """Rotation matrix -> extrinsic-xyz euler angles, matching `Rotation.as_euler("xyz")`.

    Extrinsic xyz means the matrix decomposes as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    roll = jnp.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    pitch = jnp.arctan2(
        -matrix[..., 2, 0],
        jnp.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2),
    )
    yaw = jnp.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return jnp.stack([roll, pitch, yaw], axis = -1)


def euler_xyz_to_matrix(euler: at.Float[at.Array, "*b 3"]) -> at.Float[at.Array, "*b 3 3"]:
    """Extrinsic-xyz euler angles -> rotation matrix. Inverse of `matrix_to_euler_xyz`."""
    cos_roll, sin_roll = jnp.cos(euler[..., 0]), jnp.sin(euler[..., 0])
    cos_pitch, sin_pitch = jnp.cos(euler[..., 1]), jnp.sin(euler[..., 1])
    cos_yaw, sin_yaw = jnp.cos(euler[..., 2]), jnp.sin(euler[..., 2])
    one, zero = jnp.ones_like(cos_roll), jnp.zeros_like(cos_roll)

    def _rows(*rows):
        return jnp.stack([jnp.stack(row, axis = -1) for row in rows], axis = -2)

    rot_x = _rows((one, zero, zero), (zero, cos_roll, -sin_roll), (zero, sin_roll, cos_roll))
    rot_y = _rows((cos_pitch, zero, sin_pitch), (zero, one, zero), (-sin_pitch, zero, cos_pitch))
    rot_z = _rows((cos_yaw, -sin_yaw, zero), (sin_yaw, cos_yaw, zero), (zero, zero, one))
    return rot_z @ rot_y @ rot_x


def joint_to_eef(
    actions: at.Float[at.Array, "*b 14"],
    origins: at.Float[at.Array, "6 4 4"],
    axes: at.Float[at.Array, "6 3"],
) -> at.Float[at.Array, "*b 14"]:
    """``(..., 14)`` absolute joint actions -> ``(..., 14)`` absolute EEF actions.

    Same contract as `yam_eef.joint_actions_to_eef`:
      input  ``[L joint0..5, L gripper, R joint0..5, R gripper]``
      output ``[L xyz, L rpy, L gripper, R xyz, R rpy, R gripper]``
    Grippers pass through untouched: they are commanded widths, not poses.
    """
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected trailing dim {ACTION_DIM}, got {actions.shape[-1]}")

    left_position, left_rotation = fk(actions[..., 0:JOINTS_PER_ARM], origins, axes)
    right_position, right_rotation = fk(
        actions[..., JOINTS_PER_ARM + 1 : 2 * JOINTS_PER_ARM + 1], origins, axes
    )
    return jnp.concatenate(
        [
            left_position,
            matrix_to_euler_xyz(left_rotation),
            actions[..., JOINTS_PER_ARM : JOINTS_PER_ARM + 1],
            right_position,
            matrix_to_euler_xyz(right_rotation),
            actions[..., 2 * JOINTS_PER_ARM + 1 : 2 * JOINTS_PER_ARM + 2],
        ],
        axis = -1,
    )


def apply_delta(
    actions: at.Float[at.Array, "*b ah d"],
    state: at.Float[at.Array, "*b d"],
    mask: np.ndarray,
    rpy_index_start: tuple[int, ...] | None,
) -> at.Float[at.Array, "*b ah d"]:
    """jnp equivalent of `transforms.DeltaActions`: absolute actions -> state-relative.

    Masked position dims become ``action - state``; each 3-wide extrinsic-xyz euler block
    at `rpy_index_start` becomes the euler angles of ``R_action @ R_state.inv()``. Dims
    outside both stay absolute.
    """
    subtract_mask = np.asarray(mask).copy()
    if rpy_index_start is not None:
        for start in rpy_index_start:
            subtract_mask[start : start + 3] = False
    dims = subtract_mask.shape[-1]

    offset = jnp.where(jnp.asarray(subtract_mask), state[..., :dims], 0.0)
    actions = actions.at[..., :dims].add(-offset[..., None, :])

    if rpy_index_start is not None:
        for start in rpy_index_start:
            state_rotation = euler_xyz_to_matrix(state[..., None, start : start + 3])
            action_rotation = euler_xyz_to_matrix(actions[..., start : start + 3])
            delta = action_rotation @ jnp.swapaxes(state_rotation, -1, -2)
            actions = actions.at[..., start : start + 3].set(matrix_to_euler_xyz(delta))
    return actions


def apply_absolute(
    actions: at.Float[at.Array, "*b ah d"],
    state: at.Float[at.Array, "*b d"],
    mask: np.ndarray,
    rpy_index_start: tuple[int, ...] | None,
) -> at.Float[at.Array, "*b ah d"]:
    """jnp equivalent of `transforms.AbsoluteActions`: state-relative actions -> absolute."""
    add_mask = np.asarray(mask).copy()
    if rpy_index_start is not None:
        for start in rpy_index_start:
            add_mask[start : start + 3] = False
    dims = add_mask.shape[-1]

    offset = jnp.where(jnp.asarray(add_mask), state[..., :dims], 0.0)
    actions = actions.at[..., :dims].add(offset[..., None, :])

    if rpy_index_start is not None:
        for start in rpy_index_start:
            state_rotation = euler_xyz_to_matrix(state[..., None, start : start + 3])
            delta_rotation = euler_xyz_to_matrix(actions[..., start : start + 3])
            absolute = delta_rotation @ state_rotation
            actions = actions.at[..., start : start + 3].set(matrix_to_euler_xyz(absolute))
    return actions
