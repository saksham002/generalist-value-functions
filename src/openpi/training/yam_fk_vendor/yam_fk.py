"""Forward kinematics for the I2RT YAM 6-DOF arm (ABC-130k, yam 3lego).

YAM datasets record joint space only (6 joints + 1 gripper per arm); the canonical corpus
also wants a cartesian stream. Convention (2026-07-24): the synthesized EE pose is the
**wrist flange** (``link6`` body frame) in the arm's base frame — NOT a TCP. ABC-130k mixes
three gripper types (crank/linear = 0.1347 m, flexible = 0.1 m grasp offsets) with no
per-episode type label, so any single tool offset would be wrong for a slice of the data;
the flange is the one frame defined identically for every rig.

The model is the vendored kinematics-only copy of i2rt's MJCF (vendor/yam_vendor_kin.xml,
from i2rt/robot_models/arm/yam/yam.xml on github.com/i2rt-robotics/i2rt @ main, with the
<asset> block and mesh <geom>s stripped so it compiles without the STLs — FK does not read
geometry). Chain constants (body offsets + joint axes) are read from the compiled
``MjModel``, then FK evaluates as a vectorized product of 4x4 transforms so whole
``(episodes, T, 6)`` blocks map in one call. ``fk_mujoco`` walks ``mj_kinematics`` frame by
frame instead and exists to pin the vectorized path in tests.

Both paths return the flange pose in global coordinates, which equals the arm base frame
only because this model's ``base`` body sits at the world origin. For the chunked relative
action format (``T_state^-1 @ T_action``) that distinction cancels out entirely; the flange
frame convention does not, so this xml must stay pinned.
"""

import pathlib

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

YAM_KIN_XML = pathlib.Path(__file__).resolve().parent / "vendor" / "yam_vendor_kin.xml"

_CHAIN_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6")


class YamFK:
    """Vectorized FK for one YAM arm: 6 joint angles -> flange (link6) pose in base frame."""

    def __init__(self, xml_path: pathlib.Path | str = YAM_KIN_XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        origins, axes = [], []
        for name in _CHAIN_BODIES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = self.model.body_jntadr[bid]
            assert self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
            assert np.allclose(self.model.jnt_pos[jid], 0.0), f"{name}: joint offset unsupported"
            origin = np.eye(4)
            origin[:3, :3] = Rotation.from_quat(self.model.body_quat[bid], scalar_first=True).as_matrix()
            origin[:3, 3] = self.model.body_pos[bid]
            origins.append(origin)
            axes.append(self.model.jnt_axis[jid].copy())
        self._origins = np.stack(origins)  # (6, 4, 4)
        self._axes = np.stack(axes)  # (6, 3), unit vectors

    def fk(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(..., 6)`` joint angles (rad) -> flange ``(position (..., 3), quat_xyzw (..., 4))``."""
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        assert joint_angles.shape[-1] == 6
        leading = joint_angles.shape[:-1]
        q = joint_angles.reshape(-1, 6)

        rotvecs = q[..., None] * self._axes  # (N, 6, 3)
        rot = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_matrix().reshape(-1, 6, 3, 3)
        links = np.broadcast_to(self._origins, (len(q), 6, 4, 4)).copy()
        links[:, :, :3, :3] = self._origins[None, :, :3, :3] @ rot

        flange = links[:, 0]
        for i in range(1, 6):
            flange = flange @ links[:, i]

        position = flange[:, :3, 3].reshape(*leading, 3)
        quat_xyzw = Rotation.from_matrix(flange[:, :3, :3]).as_quat().reshape(*leading, 4)
        return position, quat_xyzw

    def fk_mujoco(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reference path through ``mj_kinematics`` (one frame at a time); for validation."""
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        leading = joint_angles.shape[:-1]
        q = joint_angles.reshape(-1, 6)
        data = mujoco.MjData(self.model)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        pos = np.empty((len(q), 3))
        quat = np.empty((len(q), 4))
        for i, qi in enumerate(q):
            data.qpos[:6] = qi
            mujoco.mj_kinematics(self.model, data)
            pos[i] = data.xpos[bid]
            quat[i] = data.xquat[bid][[1, 2, 3, 0]]  # wxyz -> xyzw
        return pos.reshape(*leading, 3), quat.reshape(*leading, 4)
