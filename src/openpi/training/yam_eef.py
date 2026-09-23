"""Map YAM 14D joint actions into the 14D EEF layout the critic was pretrained on.

The YAM policy emits joint space (6 joints + 1 gripper per arm), but the critic was
pretrained on ``pos3 + euler3 + gripper`` per arm. Converting requires forward
kinematics, so this module reuses the FK implementation that produced the canonical
cartesian stream for these datasets rather than reimplementing the chain:
``rlds_dataset_builder/scratch/yam_fk.py`` (and its pinned ``vendor/yam_vendor_kin.xml``),
vendored verbatim under ``yam_fk_vendor/``.

Two conventions are inherited from that module and must not drift:
  * the synthesized pose is the **flange** (``link6``), NOT a TCP — ABC-130k mixes
    gripper types with no per-episode label, so no single tool offset is correct;
  * rotations are emitted as extrinsic-xyz euler to match
    ``transforms.DeltaActions`` / ``AbsoluteActions``, which use
    ``Rotation.from_euler("xyz", ...)``.

Convert only AFTER the policy output transform has produced absolute, unnormalized
joint actions: FK on a delta or on normalized values is meaningless.
"""

import functools
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# yam_fk.py resolves its XML relative to its own directory, so pointing at the
# directory is enough to pick up the pinned vendor model.
#
# Vendored in-repo rather than referenced at its `rlds_dataset_builder` origin because
# that is a developer-machine path: the BestOfN serve path runs on TPU hosts, where it
# does not exist. `yam_eef_jax_test` byte-compares both files against the builder's copy,
# so drift fails loudly wherever that repo is present.
DEFAULT_YAM_FK_DIR = str(pathlib.Path(__file__).resolve().parent / "yam_fk_vendor")

JOINTS_PER_ARM = 6
ACTION_DIM = 14


@functools.lru_cache(maxsize=2)
def _load_fk(yam_fk_dir: str):
    """Import and instantiate YamFK once per directory (MjModel load is not free)."""
    fk_dir = pathlib.Path(yam_fk_dir).expanduser().resolve()
    if not (fk_dir / "yam_fk.py").is_file():
        raise FileNotFoundError(f"yam_fk.py not found in {fk_dir}")
    if str(fk_dir) not in sys.path:
        sys.path.insert(0, str(fk_dir))
    from yam_fk import YamFK

    return YamFK()


def joint_actions_to_eef(actions: np.ndarray, yam_fk_dir: str = DEFAULT_YAM_FK_DIR) -> np.ndarray:
    """``(..., 14)`` absolute joint actions -> ``(..., 14)`` absolute EEF actions.

    Input  layout: ``[L joint0..5, L gripper, R joint0..5, R gripper]``
    Output layout: ``[L xyz, L rpy, L gripper, R xyz, R rpy, R gripper]``

    Grippers pass through untouched: they are commanded widths, not poses.
    """
    array = np.asarray(actions, dtype=np.float64)
    if array.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected trailing dim {ACTION_DIM}, got {array.shape[-1]}")

    fk = _load_fk(yam_fk_dir)
    leading = array.shape[:-1]

    left_position, left_quat = fk.fk(array[..., 0:JOINTS_PER_ARM])
    right_position, right_quat = fk.fk(array[..., JOINTS_PER_ARM + 1 : 2 * JOINTS_PER_ARM + 1])

    # yam_fk returns scalar-last quaternions, which is scipy's from_quat default.
    left_rpy = Rotation.from_quat(left_quat.reshape(-1, 4)).as_euler("xyz").reshape(*leading, 3)
    right_rpy = Rotation.from_quat(right_quat.reshape(-1, 4)).as_euler("xyz").reshape(*leading, 3)

    return np.concatenate(
        [
            left_position,
            left_rpy,
            array[..., JOINTS_PER_ARM : JOINTS_PER_ARM + 1],
            right_position,
            right_rpy,
            array[..., 2 * JOINTS_PER_ARM + 1 : 2 * JOINTS_PER_ARM + 2],
        ],
        axis=-1,
    ).astype(np.float32)
