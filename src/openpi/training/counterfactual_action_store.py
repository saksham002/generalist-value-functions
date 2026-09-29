"""Counterfactual Action Store: Storage and retrieval of pre-computed policy action samples.

This module provides infrastructure for storing pre-computed action samples from a
BC-trained policy for every state in an RLDS dataset, for all valid subtasks.

Actions are stored unnormalized in the dataset action space.
For each step, actions are generated for the prompt used by policy training, resulting in shape:
    [num_steps, num_samples, action_horizon, action_dim]

The store is stored as a TFDS dataset that mirrors the source RLDS layout.
This ensures identical loading behavior (file shuffling, process sharding) as the source
RLDS dataset, enabling deterministic joining via tfds.builder_from_directory().

A manifest.json at the root stores metadata about the source dataset, the policy used
to generate actions, and the action dimensions.
"""

import dataclasses
import json
import logging
from typing import Any

from etils import epath
import numpy as np

logger = logging.getLogger(__name__)

COUNTERFACTUAL_ACTION_STORE_DATASET_NAME = "counterfactual_action_store"
VERSION = "1.0.0"


def get_shard_filename(split: str, shard_idx: int) -> str:
    """Generate TFDS shard filename for counterfactual action store."""
    return f"{COUNTERFACTUAL_ACTION_STORE_DATASET_NAME}-{split}.tfrecord-{shard_idx:05d}"


# =============================================================================
# Data structures
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CounterfactualActionStoreManifest:
    """Manifest stored as manifest.json at counterfactual action store root.

    Contains metadata about the source RLDS dataset, the policy used to generate
    actions, and the resulting action dimensions. Used to validate compatibility
    between the store and the source dataset at training time.
    """

    version: str = "1.0"

    # Source RLDS identity for validation
    source_rlds_data_dir: str = ""
    source_dataset_name: str = ""
    source_dataset_version: str = ""
    source_num_episodes: int = 0

    # Action dimensions
    num_samples: int = 0
    action_dim: int = 0
    action_horizon: int = 0
    # Strided computation (stride=1 means per-timestep storage, >1 means strided)
    stride: int = 1

    # Policy used to generate actions
    policy_config_name: str = ""
    policy_checkpoint_dir: str = ""

    # Creation timestamp
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "source_rlds_data_dir": self.source_rlds_data_dir,
            "source_dataset_name": self.source_dataset_name,
            "source_dataset_version": self.source_dataset_version,
            "source_num_episodes": self.source_num_episodes,
            "num_samples": self.num_samples,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "stride": self.stride,
            "policy_config_name": self.policy_config_name,
            "policy_checkpoint_dir": self.policy_checkpoint_dir,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounterfactualActionStoreManifest":
        """Deserialize from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            source_rlds_data_dir=data.get("source_rlds_data_dir", ""),
            source_dataset_name=data.get("source_dataset_name", ""),
            source_dataset_version=data.get("source_dataset_version", ""),
            source_num_episodes=data.get("source_num_episodes", 0),
            num_samples=data.get("num_samples", 0),
            action_dim=data.get("action_dim", 0),
            action_horizon=data.get("action_horizon", 0),
            stride=data.get("stride", 1),
            policy_config_name=data.get("policy_config_name", ""),
            policy_checkpoint_dir=data.get("policy_checkpoint_dir", ""),
            created_at=data.get("created_at", ""),
        )


# =============================================================================
# Manifest I/O
# =============================================================================


def load_manifest(store_dir: str) -> CounterfactualActionStoreManifest:
    """Load and parse manifest.json from a counterfactual action store directory.

    Args:
        store_dir: Path to the store root directory.

    Returns:
        Parsed CounterfactualActionStoreManifest.

    Raises:
        FileNotFoundError: If manifest.json doesn't exist.
    """
    manifest_path = epath.Path(store_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Counterfactual action store manifest not found at {manifest_path}")

    with manifest_path.open("r") as f:
        data = json.load(f)

    return CounterfactualActionStoreManifest.from_dict(data)


def save_manifest(manifest: CounterfactualActionStoreManifest, store_dir: str) -> None:
    """Save manifest.json to a counterfactual action store directory.

    Args:
        manifest: Manifest to save.
        store_dir: Path to the store root directory.
    """
    manifest_path = epath.Path(store_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def validate_manifest_against_rlds(
    manifest: CounterfactualActionStoreManifest,
    rlds_data_dir: str,
    dataset_name: str,
    dataset_version: str,
) -> None:
    """Validate that the counterfactual action store manifest matches the source RLDS dataset.

    Args:
        manifest: The store manifest.
        rlds_data_dir: Path to the RLDS data directory.
        dataset_name: Name of the RLDS dataset.
        dataset_version: Version of the RLDS dataset.

    Raises:
        RuntimeError: If the manifest doesn't match the source dataset.
    """
    # Path validation skipped: data_dir may differ between export and training environments
    # (e.g., different cluster mount points). Name and version are sufficient for identity.
    del rlds_data_dir
    if manifest.source_dataset_name != dataset_name:
        raise RuntimeError(
            f"Counterfactual action store was created from dataset '{manifest.source_dataset_name}' "
            f"but trying to use with dataset '{dataset_name}'."
        )
    if manifest.source_dataset_version != dataset_version:
        raise RuntimeError(
            f"Counterfactual action store was created from dataset version '{manifest.source_dataset_version}' "
            f"but trying to use with version '{dataset_version}'."
        )


# =============================================================================
# TFDS-based storage (for deterministic joining with RLDS)
# =============================================================================


def get_counterfactual_action_store_tfds_feature_spec(manifest: CounterfactualActionStoreManifest) -> Any:
    """Build TFDS feature spec for counterfactual action store dataset.

    Args:
        manifest: Store manifest with dimension info.

    Returns:
        tfds.features.FeaturesDict with episode metadata and counterfactual actions tensor.
    """
    import tensorflow_datasets as tfds

    return tfds.features.FeaturesDict(
        {
            "episode_index": tfds.features.Scalar(dtype=np.int64),
            "num_steps": tfds.features.Scalar(dtype=np.int64),
            "counterfactual_actions": tfds.features.Tensor(
                shape=(None, manifest.num_samples, manifest.action_horizon, manifest.action_dim),
                dtype=np.float32,
            ),
        }
    )


class CounterfactualActionStoreTFDSShardWriter:
    """Writer that creates a single TFDS shard file matching an RLDS shard.

    Each instance writes one shard; create multiple instances for multiple shards.
    """

    def __init__(
        self,
        output_dir: str,
        shard_idx: int,
        manifest: CounterfactualActionStoreManifest,
        *,
        split: str = "train",
    ):
        """Initialize the shard writer.

        Args:
            output_dir: Root directory for the store.
            shard_idx: Index of this shard (must match source RLDS shard index).
            manifest: Manifest describing the store.
            split: Split name (e.g., "train", "val").
        """
        import tensorflow as tf

        self._output_dir = epath.Path(output_dir)
        self._shard_idx = shard_idx
        self._manifest = manifest
        self._split = split
        self._episode_count = 0
        self._total_bytes = 0

        self._dataset_dir = self._output_dir / COUNTERFACTUAL_ACTION_STORE_DATASET_NAME / VERSION
        self._dataset_dir.mkdir(parents=True, exist_ok=True)

        self._features = get_counterfactual_action_store_tfds_feature_spec(manifest)

        self._shard_path = self._dataset_dir / get_shard_filename(split, shard_idx)
        self._writer = tf.io.TFRecordWriter(str(self._shard_path))

    def write_episode(
        self,
        episode_index: int,
        num_steps: int,
        actions: np.ndarray,
    ) -> None:
        """Write a single episode's counterfactual actions to the shard.

        Args:
            episode_index: Index of the episode in the source dataset (RLDS episode_index).
            num_steps: Number of raw steps in the episode.
            actions: Counterfactual actions array [num_steps, num_samples, action_horizon, action_dim].
        """
        expected_shape_suffix = (
            self._manifest.num_samples,
            self._manifest.action_horizon,
            self._manifest.action_dim,
        )
        if actions.shape[1:] != expected_shape_suffix:
            raise ValueError(f"Expected actions shape [num_steps, {expected_shape_suffix}], got [{actions.shape}]")

        example = {
            "episode_index": episode_index,
            "num_steps": num_steps,
            "counterfactual_actions": actions.astype(np.float32),
        }

        serialized = self._features.serialize_example(example)
        self._writer.write(serialized)
        self._total_bytes += len(serialized)
        self._episode_count += 1

    def write_raw_record(self, serialized: bytes) -> None:
        """Write a pre-serialized TFRecord example verbatim.

        Used by re-sharding, which copies records between shard layouts without
        decoding/re-encoding the payload. ``serialized`` must be an example produced
        by another shard of the same store (same feature spec).
        """
        self._writer.write(serialized)
        self._total_bytes += len(serialized)
        self._episode_count += 1

    def finalize(self) -> None:
        """Close the writer."""
        self._writer.close()
        logger.info(f"Wrote {self._episode_count} episodes to shard {self._shard_idx} at {self._shard_path}")

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def shard_idx(self) -> int:
        return self._shard_idx

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def dataset_dir(self) -> epath.Path:
        return self._dataset_dir


def get_counterfactual_action_store_builder(store_dir: str) -> Any:
    """Load counterfactual action store as a TFDS builder.

    Args:
        store_dir: Path to the store root directory.

    Returns:
        TFDS DatasetBuilder for the store.

    Raises:
        FileNotFoundError: If no TFDS dataset found in the store directory.
    """
    import tensorflow_datasets as tfds

    store_path = epath.Path(store_dir)

    expected_tfds_dir = store_path / COUNTERFACTUAL_ACTION_STORE_DATASET_NAME / VERSION
    if expected_tfds_dir.exists() and (expected_tfds_dir / "dataset_info.json").exists():
        return tfds.builder_from_directory(str(expected_tfds_dir))

    first_candidate = next(store_path.glob("*/*/dataset_info.json"), None)
    if first_candidate is None:
        raise FileNotFoundError(
            f"No TFDS dataset found in {store_dir}. "
            f"Expected dataset at {expected_tfds_dir} or a dataset_info.json somewhere under {store_path}."
        )

    dataset_dir = first_candidate.parent
    logger.info(f"Found counterfactual action store TFDS dataset at {dataset_dir}")
    return tfds.builder_from_directory(str(dataset_dir))


# =============================================================================
# Strided action expansion
# =============================================================================


def expand_strided_counterfactual_actions(
    strided_actions,
    stride: int,
    num_steps: int,
    target_action_horizon: int,
):
    """Expand strided counterfactual actions to per-timestep format.

    For timestep t:
    - strided_idx = t // stride
    - offset = t % stride
    - result[t] = strided_actions[strided_idx, :, :, offset:offset+target_action_horizon, :]

    Args:
        strided_actions: Tensor of shape [num_strided_positions, num_samples, stored_action_horizon, action_dim]
        stride: Stride used during computation
        num_steps: Number of timesteps in the trajectory
        target_action_horizon: Desired output action horizon (must satisfy constraint:
                               (stride-1) + target_action_horizon <= stored_action_horizon)

    Returns:
        Expanded actions tensor [num_steps, num_samples, target_action_horizon, action_dim]
    """
    import tensorflow as tf

    # strided_indices[t] = t // stride (which strided position to use for timestep t)
    strided_indices = tf.range(num_steps) // stride
    # gathered: [num_steps, num_samples, stored_action_horizon, action_dim]
    gathered = tf.gather(strided_actions, strided_indices)

    # offsets[t] = t % stride (offset into the action_horizon dimension)
    offsets = tf.range(num_steps) % stride
    # action_indices[t, h] = offset[t] + h, for h in [0, target_action_horizon)
    action_indices = offsets[:, None] + tf.range(target_action_horizon)[None, :]  # [num_steps, target_action_horizon]

    # Transpose to [num_steps, stored_action_horizon, num_samples, action_dim]
    # so we can use batch_dims=1 to gather along stored_action_horizon
    gathered_transposed = tf.transpose(gathered, [0, 2, 1, 3])
    # After gather: [num_steps, target_action_horizon, num_samples, action_dim]
    result_transposed = tf.gather(gathered_transposed, action_indices, batch_dims=1)
    # Transpose back to [num_steps, num_samples, target_action_horizon, action_dim]
    return tf.transpose(result_transposed, [0, 2, 1, 3])
