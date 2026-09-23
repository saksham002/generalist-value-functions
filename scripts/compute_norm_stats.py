"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

from etils import epath
import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.rlds_dataset as rlds_dataset
import openpi.transforms as transforms

# Chunk length used for action_diff per-step stats. Sized to cover the longest
# action_horizon any downstream consumer might use (50 for the RoboCOIN
# pretraining configs); shorter consumers slice
# the resulting (50, D) array down to their own horizon at config-create time.
ACTION_DIFF_HORIZON = 50


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None:
        # Training-mode RLDS streams infinitely → len(dataset) raises. Use max_frames
        # directly to compute batch count.
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    fine_tune: str | None = None,
    max_frames: int | None = 200_000,
    dataset_name: str | None = None,
    output_dir: str | None = None,
    data_dir: str | None = None,
):
    """Compute normalization statistics for a config.

    Args:
        config_name: name of a registered TrainConfig.
        fine_tune: optional name of a registered FineTuneConfig. When provided, its
            data/model overrides are applied to the base TrainConfig before computing
            stats — so the stats reflect the dataset the fine-tune actually trains on
            (e.g. task data when fine-tuning a RoboCOIN base).
        max_frames: cap on number of frames to iterate. Defaults to 200_000 — enough
            for stable mean/std/quantile estimates without relying on the RLDS
            dataset length.
        dataset_name: optional "name:version" or "name" string. When set, replaces
            the data factory's `datasets` field with a single-element tuple referring
            to this RLDS dataset. Useful for computing per-task norm stats without
            registering a new TrainConfig per task.
        output_dir: optional explicit directory to write `norm_stats.json` to.
            Overrides the default `config.assets_dirs / repo_id` path.
        data_dir: optional override for the RLDS root (``config.data.rlds_data_dir``).
            Useful for computing stats against a local mirror of the GCS dataset
            without modifying the registered config.
    """
    config = _config.get_config(config_name)
    if fine_tune is not None:
        ft_config = _config.get_fine_tune_config(fine_tune)
        config = ft_config.apply_overrides(config)
        print(f"Applied FineTuneConfig overrides from '{fine_tune}'")

    if data_dir is not None:
        config = dataclasses.replace(
            config, data = dataclasses.replace(config.data, rlds_data_dir = data_dir)
        )
        print(f"Overrode rlds_data_dir to {data_dir}")

    if dataset_name is not None:
        if ":" in dataset_name:
            ds_name, ds_version = dataset_name.split(":", 1)
        else:
            ds_name, ds_version = dataset_name, "1.0.0"
        new_dataset = rlds_dataset.RLDSDataset(name = ds_name, version = ds_version, weight = 1.0)
        config = dataclasses.replace(
            config, data = dataclasses.replace(config.data, datasets = (new_dataset,))
        )
        print(f"Overrode dataset to {ds_name}:{ds_version}")

    # The script computes two flavors of action stats: absolute action[0] (1-D) and
    # the chunk-wise-delta of the whole chunk (2-D, per-timestep). Force the data
    # pipeline to emit absolute actions so the DeltaActions transform applied here
    # is the only source of deltas — even if the config has use_chunk_wise_delta=True
    # for training.
    if getattr(config.data, "use_chunk_wise_delta", False):
        config = dataclasses.replace(
            config, data = dataclasses.replace(config.data, use_chunk_wise_delta = False),
        )
        print("Overrode use_chunk_wise_delta=False for stats computation.")

    # Stats are computed from state/actions only, never from counterfactual actions. Keeping
    # the store attached would make this fail on a config whose store is still being built —
    # which is the normal ordering, since caching consumes a policy trained on these stats.
    if getattr(config.data, "counterfactual_action_store_dir", None) is not None:
        config = dataclasses.replace(
            config, data = dataclasses.replace(config.data, counterfactual_action_store_dir = None),
        )
        print("Overrode counterfactual_action_store_dir=None for stats computation.")

    # Stats must be computed at the source 60 Hz cadence: _slice_action_diff_norm_stats
    # applies the [1::2] stride when loading, mapping 30 Hz chunk index t to native 2t+1.
    if getattr(config.data, "subsample", False):
        config = dataclasses.replace(config, data = dataclasses.replace(config.data, subsample = False))
        print("Overrode subsample=False for stats computation.")

    data_config = config.data.create(config.assets_dirs, config.model)

    # Pull chunks at the config's effective action_horizon (after FineTune overrides)
    # so the (H, D) action_diff_stats array covers the consumer's horizon. Falls back
    # to ACTION_DIFF_HORIZON for configs without an explicit action_horizon. Shorter
    # consumers slice down at data_config.create() time.
    action_horizon = config.action_horizon if config.action_horizon is not None else ACTION_DIFF_HORIZON

    if data_config.rlds_data_dir is None:
        raise ValueError("Norm stats are computed from RLDS datasets; the config must set rlds_data_dir.")
    data_loader, num_batches = create_rlds_dataloader(data_config, action_horizon, config.batch_size, max_frames)
    output_id = data_config.repo_id

    # 14D bimanual chunk-wise-delta action space; the mask keeps grippers absolute.
    # The EEF layout (left xyz, left rpy, left gripper, right xyz, right rpy, right
    # gripper) composes the two rpy blocks as relative rotations, whereas a joint
    # layout is a plain subtraction — so mirror the data config's use_eef here,
    # otherwise the stats describe a different delta than training consumes.
    use_eef = getattr(config.data, "use_eef", True)
    action_diff_transform = transforms.DeltaActions(
        mask = transforms.make_bool_mask(6, -1, 6, -1),
        rpy_index_start = (3, 10) if use_eef else None,
    )
    print(f"action_diff delta uses use_eef={use_eef}")

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()  # populated only with action[0] -> (D,)
    action_diff_stats: list[normalize.RunningStats] = []  # one RunningStats per timestep -> (H, D)

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        state = np.asarray(batch["state"])      # (B, D_state)
        actions = np.asarray(batch["actions"])  # (B, H, D_act)
        state_stats.update(state)
        # Use only the first action of the chunk for the absolute-action stats.
        action_stats.update(actions[:, 0, :])

        # Apply DeltaActions to the absolute action chunk to get per-step deltas.
        delta = action_diff_transform({"state": state, "actions": actions})["actions"]  # (B, H, D_act)
        if not action_diff_stats:
            action_diff_stats = [normalize.RunningStats() for _ in range(delta.shape[1])]
        for t in range(delta.shape[1]):
            action_diff_stats[t].update(delta[:, t, :])

    # Stack per-timestep stats into (H, D) arrays so action_diff is 2-D, not collapsed.
    diff_per_t = [s.get_statistics() for s in action_diff_stats]
    action_diff_norm = normalize.NormStats(
        mean = np.stack([s.mean for s in diff_per_t], axis = 0),
        std = np.stack([s.std for s in diff_per_t], axis = 0),
        q01 = np.stack([s.q01 for s in diff_per_t], axis = 0),
        q99 = np.stack([s.q99 for s in diff_per_t], axis = 0),
    )

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
        "action_diff": action_diff_norm,
    }

    if output_dir is not None:
        output_path = epath.Path(output_dir)
    else:
        output_path = config.assets_dirs / output_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
