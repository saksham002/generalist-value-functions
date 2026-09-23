"""Generic RLDS build/partition helpers shared by the offline build scripts.

Config resolution, episode counting, episode-index extraction, and worker
partition parsing used by the counterfactual-action-store build pipeline.
"""

import logging

logger = logging.getLogger(__name__)


def resolve_config(config_name: str):
    """Resolve openpi config and return (config, data_config, dataset_cfg, model_config)."""
    import openpi.training.config as _config

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is None:
        raise ValueError("Config must have rlds_data_dir set.")

    datasets = data_config.datasets
    if not datasets:
        raise ValueError("Config must have datasets configured.")
    if len(datasets) > 1:
        logger.warning("Multiple datasets configured, only processing first one.")

    return config, data_config, datasets[0], config.model


def get_total_episodes(data_dir: str, dataset_cfg, split: str) -> int:
    """Get total episode count from source dataset."""
    import tensorflow_datasets as tfds

    builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)
    return builder.info.splits[split].num_examples


def get_rlds_episode_index(episode: dict) -> int:
    """Extract the episode_index feature from RLDS episode.

    For RoboCOIN, this is in each step's metadata as 'episode_index' (int64).
    This is the ground-truth identifier, NOT the positional index.
    """
    first_step = episode["steps"][0]
    return int(first_step["episode_index"])


def parse_partition_split(partition_split: str, num_workers: int) -> list[str]:
    """Parse partition_split string into per-worker partition assignments.

    Args:
        partition_split: Format 'partition1:count1,partition2:count2,...'
        num_workers: Total number of workers (must match sum of counts)

    Returns:
        List of partition names, one per worker_id.
    """
    assignments: list[str] = []
    for raw_part in partition_split.split(","):
        part = raw_part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid partition_split format: '{part}'. Expected 'partition:count'.")
        partition, count_str = part.split(":", 1)
        count = int(count_str)
        assignments.extend([partition] * count)

    if len(assignments) != num_workers:
        raise ValueError(
            f"partition_split counts sum to {len(assignments)}, but num_workers={num_workers}. These must match."
        )
    return assignments
