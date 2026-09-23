"""Generic base RLDS dataset loader.

Extracts common RLDS loading logic (dataset building, action chunking, flattening,
shuffling, batching) into a reusable base class.
Subclasses override trajectory_transforms() and RL hooks to customize dataset-specific
processing.

Supports two modes:
- Training mode (default): Infinite streaming with flattening, shuffling, and batching.
- Trajectory mode (return_trajectories=True): Finite dataset yielding full trajectories
  for validation, with frame transforms applied per-timestep via traj_map.
"""

from collections.abc import Sequence
import dataclasses
import logging
from typing import Any

import openpi.training.counterfactual_action_store as counterfactual_action_store


@dataclasses.dataclass
class RLDSDataset:
    """Specification for a single RLDS dataset to load."""

    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class BaseRldsDataset:
    """Generic RLDS dataset loader with customization hooks.

    Handles the common RLDS pipeline: loading via dlimp/tfds, repeating for infinite
    streaming, action chunking, flattening trajectories to frames, combining multiple
    datasets with weighted sampling, shuffle buffering, and batching.

    Supports multi-host training by sharding the dataset across JAX processes.

    Subclasses override:
        trajectory_transforms(traj, dataset_cfg): Per-trajectory processing after loading.
        _get_per_step_done_flags(raw_traj, mapped_traj, per_step_rewards): Optional RL done semantics.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,
        split: str = "train",
        shuffle: bool = True,
        shuffle_seed: int = 86,
        action_chunk_size: int = 16,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        critic_mode: bool = False,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        image_obs_keys: Sequence[str] = (),
        image_size: tuple[int, int] | None = None,
        include_images: bool = True,
        decode_images: bool = True,
        return_trajectories: bool = False,
        max_trajectories: int | None = None,
        max_num_demos: int | None = None,
        counterfactual_action_store_dir: str | None = None,
        repeat_dataset: bool = True,
    ):
        if max_num_demos is not None and max_num_demos <= 0:
            raise ValueError(f"max_num_demos must be a positive integer, got {max_num_demos}")
        self._max_num_demos = max_num_demos

        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import jax
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        self._num_parallel_calls = num_parallel_calls
        self._num_parallel_reads = num_parallel_reads
        self._action_chunk_size = action_chunk_size
        self._critic_mode = critic_mode
        self._discount = discount
        self._reward_scale = reward_scale
        self._reward_bias = reward_bias
        self._include_images = include_images
        self._image_obs_keys = tuple(image_obs_keys) if include_images else ()
        self._image_size = image_size
        self._decode_images = decode_images
        self._return_trajectories = return_trajectories
        self._max_trajectories = max_trajectories
        self._counterfactual_action_store_dir = counterfactual_action_store_dir
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed

        # Load counterfactual action store manifest if configured
        self._ca_manifest: counterfactual_action_store.CounterfactualActionStoreManifest | None = None
        if counterfactual_action_store_dir is not None:
            self._ca_manifest = counterfactual_action_store.load_manifest(counterfactual_action_store_dir)
            logging.info(
                f"Loaded counterfactual action store manifest: {self._ca_manifest.source_dataset_name} "
                f"v{self._ca_manifest.source_dataset_version}, "
                f"num_samples={self._ca_manifest.num_samples}, "
                f"action_horizon={self._ca_manifest.action_horizon}, "
                f"action_dim={self._ca_manifest.action_dim}"
            )

        # For multi-host training, each process loads its own shard
        process_count = jax.process_count()
        process_index = jax.process_index()

        if process_count > 1 and not return_trajectories:
            assert batch_size % process_count == 0, (
                f"batch_size ({batch_size}) must be divisible by process_count ({process_count})"
            )
            local_batch_size = batch_size // process_count
            logging.info(
                f"Multi-host RLDS loading: process {process_index}/{process_count}, local_batch_size={local_batch_size}"
            )
        else:
            local_batch_size = batch_size
            if process_count > 1 and return_trajectories:
                logging.info(
                    f"Multi-host RLDS loading (trajectory mode): process {process_index}/{process_count}"
                )

        def prepare_single_dataset(dataset_cfg: RLDSDataset, *, for_trajectories: bool):
            """Prepare a single dataset for either training or trajectory mode.

            Args:
                dataset_cfg: Dataset configuration.
                for_trajectories: If True, prepare for trajectory mode (finite, yields trajectories).
                                  If False, prepare for training (infinite, flattened frames).
            """
            builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)

            if for_trajectories:
                # Use strided sampling for validation to get diverse trajectories across the dataset.
                # TFDS doesn't support stride syntax, so we build a union of individual indices.
                if max_trajectories is not None and self._counterfactual_action_store_dir is not None:
                    raise ValueError(
                        "max_trajectories with counterfactual_action_store_dir is not supported in trajectory mode: "
                        "strided episode selection would misalign the zip between RLDS and CA store datasets."
                    )
                if max_trajectories is not None:
                    num_episodes = builder.info.splits[split].num_examples
                    stride = max(1, num_episodes // max_trajectories)
                    indices = list(range(0, num_episodes, stride))[:max_trajectories]
                    base_split = "+".join(f"{split}[{i}:{i + 1}]" for i in indices)
                else:
                    base_split = split
                # Shard trajectory-mode datasets across JAX processes the same
                # way training does (see ``not for_trajectories`` branch
                # below). Without this, every host iterates the full N-shard
                # val dataset which accumulates host RAM until at least one
                # worker is OOM-killed and drops the JAX coordinator.
                # Exception: when the split has fewer examples than processes,
                # even-splitting hands the trailing hosts an empty slice and TFDS
                # raises "Instruction [] corresponds to no data!". Such small
                # splits don't risk the OOM above, so every host reads the full
                # split instead.
                if process_count > 1 and builder.info.splits[split].num_examples >= process_count:
                    split_to_use = tfds.split_for_jax_process(
                        base_split, process_index=process_index, process_count=process_count
                    )
                else:
                    split_to_use = base_split
                logging.info(
                    f"  Dataset {dataset_cfg.name} (trajectory mode): using split {split_to_use!r}"
                )
                do_shuffle = False
            else:
                # Apply max_num_demos limit if specified (uses TFDS absolute-count slicing)
                base_split = f"{split}[:{self._max_num_demos}]" if self._max_num_demos is not None else split
                split_to_use = (
                    tfds.split_for_jax_process(base_split, process_index=process_index, process_count=process_count)
                    if process_count > 1
                    else base_split
                )
                logging.info(f"  Dataset {dataset_cfg.name}: using split {split_to_use!r}")
                do_shuffle = shuffle

            read_config_kwargs = {"shuffle_seed": shuffle_seed} if do_shuffle else None

            dataset = dl.DLataset.from_rlds(
                builder,
                split=split_to_use,
                shuffle=do_shuffle,
                num_parallel_reads=num_parallel_reads,
                read_config_kwargs=read_config_kwargs,
            )

            # Join with counterfactual action store if configured. The join fires for
            # any split; _join_counterfactual_action_store skips gracefully when the
            # store has no shards for the requested split (e.g. a train-only store
            # queried for "val"), so this is safe for stores that don't cover a split.
            if self._counterfactual_action_store_dir is not None:
                dataset = self._join_counterfactual_action_store(
                    dataset,
                    dataset_cfg,
                    split,
                    process_index if process_count > 1 else 0,
                    max(1, process_count),
                )

            # Episode-level filtering runs after the counterfactual join (dropping episodes
            # before it would misalign the zip) and before repeat(), so a dropped episode
            # never reaches the frame pipeline in either trajectory or training mode.
            dataset = dataset.filter(lambda traj: self.trajectory_filter(traj))

            if not for_trajectories and repeat_dataset:
                dataset = dataset.repeat()

            dataset = dataset.traj_map(lambda traj: self._prepare_trajectory(traj, dataset_cfg), num_parallel_calls)
            dataset = dataset.traj_map(lambda traj: self._chunk_actions(traj, action_chunk_size), num_parallel_calls)

            if for_trajectories:
                # Apply frame transforms to each step within the trajectory
                dataset = dataset.traj_map(
                    lambda traj: self._apply_frame_transforms_to_trajectory(traj), num_parallel_calls
                )
            else:
                dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
                dataset = dataset.frame_map(lambda frame: self.frame_transforms(frame), num_parallel_calls)
                dataset = dataset.filter(lambda frame: self.frame_filter(frame))
            return dataset

        logging.info(f"Preparing {len(datasets)} datasets (split={split}, trajectory_mode={return_trajectories})...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)

        if return_trajectories:
            # Trajectory mode: dataset yields full trajectories with frame transforms applied
            all_datasets = [prepare_single_dataset(dataset_cfg, for_trajectories=True) for dataset_cfg in datasets]
            # Concatenate datasets (for trajectory mode we don't sample, just concatenate)
            if len(all_datasets) == 1:
                final_dataset = all_datasets[0]
            else:
                final_dataset = all_datasets[0]
                for ds in all_datasets[1:]:
                    final_dataset = final_dataset.concatenate(ds)
            self.dataset = final_dataset
            logging.info("Dataset prepared in trajectory mode (lazy evaluation)")
        else:
            # Training mode: streaming with flattening
            all_datasets = [prepare_single_dataset(dataset_cfg, for_trajectories=False) for dataset_cfg in datasets]
            weights = [dataset.weight for dataset in datasets]
            final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights = weights)
            if shuffle:
                final_dataset = final_dataset.shuffle(shuffle_buffer_size)
            final_dataset = final_dataset.batch(local_batch_size)
            final_dataset = final_dataset.with_ram_budget(1)
            self.dataset = final_dataset

        self.batch_size = batch_size  # Global batch size
        self.local_batch_size = local_batch_size
        self.shuffle = shuffle

    def _join_counterfactual_action_store(
        self,
        dataset,
        dataset_cfg: RLDSDataset,
        split: str,
        process_index: int,
        process_count: int,
    ):
        """Join counterfactual action store episodes with RLDS episodes at trajectory level.

        Uses identical TFDS loading settings as the RLDS dataset to ensure deterministic
        episode ordering for the zip.

        Args:
            dataset: The RLDS DLataset to join with.
            dataset_cfg: Dataset configuration (for validation).
            split: Split name (e.g., "train").
            process_index: This process's index for sharding.
            process_count: Total number of processes.

        Returns:
            Joined dataset where each trajectory has counterfactual actions attached.
        """
        import tensorflow as tf
        import tensorflow_datasets as tfds

        if self._ca_manifest is None:
            return dataset

        counterfactual_action_store.validate_manifest_against_rlds(
            self._ca_manifest,
            rlds_data_dir="",
            dataset_name=dataset_cfg.name,
            dataset_version=dataset_cfg.version,
        )

        # Validate strided action constraint once at init (not per-episode)
        if self._ca_manifest.stride > 1:
            max_offset = self._ca_manifest.stride - 1
            stored_action_horizon = self._ca_manifest.action_horizon
            required_horizon = max_offset + self._action_chunk_size
            if required_horizon > stored_action_horizon:
                raise ValueError(
                    f"Strided CA constraint violated: (stride-1) + target_action_horizon <= stored_action_horizon. "
                    f"Got: {max_offset} + {self._action_chunk_size} = {required_horizon} > {stored_action_horizon}"
                )

        ca_builder = counterfactual_action_store.get_counterfactual_action_store_builder(
            self._counterfactual_action_store_dir
        )

        # A store need not cover every RLDS split (e.g. a train-only store queried for
        # "val"); skip the join rather than failing on a missing split.
        if split not in ca_builder.info.splits:
            logging.warning(
                "Counterfactual action store at %s has no '%s' split (available: %s); skipping join.",
                self._counterfactual_action_store_dir, split, list(ca_builder.info.splits),
            )
            return dataset

        # Mirror the RLDS trajectory-load guard: when the split has fewer examples
        # than processes, even-splitting hands trailing hosts empty slices and (worse)
        # the RLDS side reads the FULL split while this side would read a per-host
        # slice — desyncing the positional zip (episode-index mismatch). Read the full
        # split here too so both sides enumerate episodes in the same order.
        if process_count > 1 and ca_builder.info.splits[split].num_examples >= process_count:
            split_to_use = tfds.split_for_jax_process(
                split, process_index=process_index, process_count=process_count
            )
        else:
            split_to_use = split

        read_config = tfds.ReadConfig(
            skip_prefetch=True,
            shuffle_seed=self._shuffle_seed if self._shuffle else None,
            num_parallel_calls_for_interleave_files=self._num_parallel_reads,
            interleave_cycle_length=self._num_parallel_reads,
        )

        ca_ds = ca_builder.as_dataset(
            split=split_to_use,
            shuffle_files=self._shuffle,
            read_config=read_config,
        )

        deterministic_options = tf.data.Options()
        deterministic_options.deterministic = True
        dataset = dataset.with_options(deterministic_options)
        ca_ds = ca_ds.with_options(deterministic_options)

        zipped = tf.data.Dataset.zip((dataset, ca_ds))

        def merge_episode(rlds_episode: dict, ca_episode: dict) -> dict:
            rlds_ep_idx = rlds_episode["episode_index"][0]
            ca_ep_idx = ca_episode["episode_index"]

            tf.debugging.assert_equal(
                rlds_ep_idx,
                ca_ep_idx,
                message="Episode index mismatch! RLDS and counterfactual action store are out of sync.",
            )

            counterfactual_actions = ca_episode["counterfactual_actions"]

            # Expand strided actions if stride > 1
            if self._ca_manifest.stride > 1:
                traj_len = rlds_episode["_len"][0]
                counterfactual_actions = counterfactual_action_store.expand_strided_counterfactual_actions(
                    counterfactual_actions,
                    stride=self._ca_manifest.stride,
                    num_steps=traj_len,
                    target_action_horizon=self._action_chunk_size,
                )

            # Store counterfactual actions in trajectory
            rlds_episode["counterfactual_actions"] = counterfactual_actions

            # Broadcast scalar metadata to trajectory length for dlimp flatten
            traj_len = rlds_episode["_len"][0]
            rlds_episode["_ca_episode_index"] = tf.repeat(ca_episode["episode_index"], traj_len)

            return rlds_episode

        import dlimp as dl

        merged = zipped.map(merge_episode, num_parallel_calls=self._num_parallel_calls)
        return dl.DLataset.from_tfds_dataset(merged, is_flattened=False)

    def trajectory_transforms(self, traj: dict, dataset_cfg: RLDSDataset) -> dict:
        """Per-trajectory processing. Override in subclasses.

        Default implementation extracts standard RLDS fields.
        Must produce a dict with at least an "actions" key for action chunking.
        """
        return {
            "actions": traj["action"],
            "observation": traj.get("observation", {}),
            "prompt": traj.get("language_instruction", ""),
        }

    def _prepare_trajectory(self, raw_traj: dict, dataset_cfg: RLDSDataset) -> dict:
        mapped_traj = self.trajectory_transforms(raw_traj, dataset_cfg)
        if "actions" not in mapped_traj:
            raise ValueError("trajectory_transforms must produce an 'actions' key for action chunking.")

        # Carry over counterfactual action data if present
        if "counterfactual_actions" in raw_traj:
            mapped_traj["counterfactual_actions"] = raw_traj["counterfactual_actions"]
        for key in ("_ca_episode_index",):
            if key in raw_traj:
                mapped_traj[key] = raw_traj[key]

        if self._critic_mode:
            return self._apply_rl_fields(raw_traj, mapped_traj, self._action_chunk_size)
        return mapped_traj

    def _apply_rl_fields(self, raw_traj: dict, mapped_traj: dict, action_chunk_size: int) -> dict:
        import tensorflow as tf

        per_step_rewards = self._get_per_step_rewards(raw_traj, mapped_traj)
        transformed_rewards = per_step_rewards * self._reward_scale + self._reward_bias
        per_step_terminations, per_step_truncations = self._get_per_step_done_flags(
            raw_traj, mapped_traj, per_step_rewards
        )
        per_step_dones = tf.logical_or(per_step_terminations, per_step_truncations)
        mc_returns = self._compute_mc_returns_tf(
            transformed_rewards, per_step_dones, self._discount, raw_traj=raw_traj, mapped_traj=mapped_traj
        )

        traj_len = tf.shape(mapped_traj["actions"])[0]
        chunk_rewards, chunk_terminations, chunk_truncations = self._compute_chunk_rl_fields(
            transformed_rewards, per_step_terminations, per_step_truncations, action_chunk_size, traj_len
        )

        next_indices = tf.minimum(tf.range(traj_len) + action_chunk_size, traj_len - 1)
        observation = mapped_traj.get("observation", {})
        if not isinstance(observation, dict):
            raise ValueError("trajectory_transforms must produce observation as a dict when critic_mode=True.")
        next_observation = {key: tf.gather(value, next_indices) for key, value in observation.items()}
        next_actions_raw = tf.gather(mapped_traj["actions"], next_indices)

        mapped_traj["reward"] = chunk_rewards
        mapped_traj["mc_return"] = mc_returns
        mapped_traj["termination"] = chunk_terminations
        mapped_traj["truncation"] = chunk_truncations
        mapped_traj["next_observation"] = next_observation
        mapped_traj["next_actions_raw"] = next_actions_raw
        if "counterfactual_actions" in mapped_traj:
            mapped_traj["counterfactual_next_actions"] = tf.gather(mapped_traj["counterfactual_actions"], next_indices)

        return mapped_traj

    def _get_per_step_rewards(self, raw_traj: dict, mapped_traj: dict):
        del mapped_traj
        import tensorflow as tf

        if "reward" not in raw_traj:
            available_keys = ", ".join(sorted(raw_traj.keys()))
            raise ValueError(
                "Could not derive per-step rewards from trajectory: expected key 'reward'. "
                f"Available keys: {available_keys}"
            )
        return tf.cast(raw_traj["reward"], tf.float32)

    def _get_per_step_done_flags(self, raw_traj: dict, mapped_traj: dict, per_step_rewards):
        del mapped_traj
        del per_step_rewards
        import tensorflow as tf

        termination = self._find_first_bool_tensor(raw_traj, ("is_terminal", "terminal", "termination"))
        if termination is None:
            available_keys = ", ".join(sorted(raw_traj.keys()))
            raise ValueError(
                "Could not derive per-step termination flags from trajectory. Expected one of "
                "{is_terminal, terminal, termination}. "
                f"Available keys: {available_keys}"
            )
        truncation = self._find_first_bool_tensor(raw_traj, ("is_truncated", "truncated", "truncation"))
        if truncation is None:
            is_last = self._find_first_bool_tensor(raw_traj, ("is_last",))
            if is_last is None:
                available_keys = ", ".join(sorted(raw_traj.keys()))
                raise ValueError(
                    "Could not derive per-step truncation flags from trajectory. Expected one of "
                    "{is_truncated, truncated, truncation} or fallback key {is_last}. "
                    f"Available keys: {available_keys}"
                )
            truncation = tf.logical_and(is_last, tf.logical_not(termination))
        return termination, truncation

    def _find_first_bool_tensor(self, traj: dict, keys: Sequence[str]):
        import tensorflow as tf

        for key in keys:
            if key in traj:
                return tf.cast(traj[key], tf.bool)
        return None

    def _chunk_indices(self, traj_len, chunk_size):
        import tensorflow as tf

        base_indices = tf.broadcast_to(
            tf.range(chunk_size)[None],
            [traj_len, chunk_size],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, chunk_size],
        )
        valid_mask = base_indices < traj_len
        clamped_indices = tf.minimum(base_indices, traj_len - 1)
        return clamped_indices, valid_mask

    def _chunk_actions(self, traj: dict, action_chunk_size: int) -> dict:
        """Chunk action sequences and optional next_actions_raw."""
        import tensorflow as tf

        traj_len = tf.shape(traj["actions"])[0]
        action_chunk_indices, _ = self._chunk_indices(traj_len, action_chunk_size)
        traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
        if "next_actions_raw" in traj:
            traj["next_actions"] = tf.gather(traj["next_actions_raw"], action_chunk_indices)
            del traj["next_actions_raw"]
        return traj

    def _compute_chunk_rl_fields(self, rewards, terminations, truncations, chunk_size, traj_len):
        import tensorflow as tf

        chunk_indices, valid_mask = self._chunk_indices(traj_len, chunk_size)
        valid_mask_float = tf.cast(valid_mask, tf.float32)

        # Compute done flags and mask rewards after termination/truncation
        chunk_dones_raw = tf.gather(tf.logical_or(terminations, truncations), chunk_indices)
        chunk_dones_masked = tf.logical_and(chunk_dones_raw, valid_mask)
        chunk_dones_float = tf.cast(chunk_dones_masked, tf.float32)

        # Exclusive cumsum: cumsum[j] = sum of done[0:j-1]
        # If any done occurred before position j, this will be > 0
        cumsum_dones = tf.cumsum(chunk_dones_float, axis=1, exclusive=True)
        not_after_done_mask = tf.cast(tf.equal(cumsum_dones, 0.0), tf.float32)

        # Combine validity mask with termination mask
        reward_mask = valid_mask_float * not_after_done_mask

        chunk_rewards_raw = tf.gather(rewards, chunk_indices)
        discount_weights = tf.pow(self._discount, tf.cast(tf.range(chunk_size), tf.float32))
        chunk_rewards_masked = chunk_rewards_raw * reward_mask
        chunk_rewards = tf.reduce_sum(chunk_rewards_masked * discount_weights[None, :], axis=1)

        chunk_term_raw = tf.gather(terminations, chunk_indices)
        chunk_trunc_raw = tf.gather(truncations, chunk_indices)
        chunk_term_masked = tf.logical_and(chunk_term_raw, valid_mask)
        chunk_trunc_masked = tf.logical_and(chunk_trunc_raw, valid_mask)
        chunk_terminations = tf.reduce_any(chunk_term_masked, axis=1)
        chunk_truncations = tf.reduce_any(chunk_trunc_masked, axis=1)

        return chunk_rewards, chunk_terminations, chunk_truncations

    def _compute_mc_returns_tf(self, rewards, dones, discount, raw_traj=None, mapped_traj=None):
        del raw_traj, mapped_traj
        import tensorflow as tf

        rewards_rev = tf.reverse(rewards, axis=[0])
        dones_rev = tf.reverse(tf.cast(dones, tf.float32), axis=[0])

        def scan_fn(accumulated, inputs):
            reward, done = inputs
            return reward + discount * (1.0 - done) * accumulated

        mc_returns_rev = tf.scan(scan_fn, (rewards_rev, dones_rev), initializer=tf.constant(0.0, dtype=tf.float32))
        return tf.reverse(mc_returns_rev, axis=[0])

    def frame_transforms(self, frame: dict) -> dict:
        """Apply per-frame transforms. Override in subclasses for custom transforms.

        Default implementation decodes images from bytes to uint8 arrays. When
        `decode_images=False` is set on the dataset, raw image bytes are passed
        through unchanged so a downstream consumer (e.g. validation cache) can
        decode lazily and avoid holding decoded multi-MB arrays in host RAM.
        """
        if not self._decode_images:
            return frame

        import tensorflow as tf

        for key in self._image_obs_keys:
            if "observation" not in frame or key not in frame["observation"]:
                raise ValueError(
                    f"Configured image observation key '{key}' not found under frame['observation'] during decoding."
                )
            image = tf.io.decode_image(
                frame["observation"][key], expand_animations = False, dtype = tf.uint8
            )
            if self._image_size is not None:
                # Resize here to avoid batching errors when a dataset contains different image sizes.
                image = tf.image.resize(
                    image,
                    self._image_size,
                    method = tf.image.ResizeMethod.BILINEAR,
                    antialias = True,
                )
                image = tf.cast(tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8)
            frame["observation"][key] = image

        if "next_observation" in frame:
            for key in self._image_obs_keys:
                if key not in frame["next_observation"]:
                    raise ValueError(
                        f"Configured image observation key '{key}' not found under frame['next_observation'] during "
                        "decoding."
                    )
                image = tf.io.decode_image(
                    frame["next_observation"][key], expand_animations = False, dtype = tf.uint8
                )
                if self._image_size is not None:
                    # Resize here to avoid batching errors when a dataset contains different image sizes.
                    image = tf.image.resize(
                        image,
                        self._image_size,
                        method = tf.image.ResizeMethod.BILINEAR,
                        antialias = True,
                    )
                    image = tf.cast(tf.clip_by_value(tf.round(image), 0.0, 255.0), tf.uint8)
                frame["next_observation"][key] = image

        return frame

    def frame_filter(self, frame: dict) -> bool:
        """Filter predicate for frames. Override in subclasses to filter out invalid frames.

        Returns True to keep the frame, False to discard it.
        Default implementation keeps all frames.
        """
        import tensorflow as tf

        return tf.constant(value=True)

    def trajectory_filter(self, traj: dict) -> bool:
        """Filter predicate for whole episodes. Override in subclasses to drop episodes.

        Runs on the raw trajectory, before ``trajectory_transforms``, so episode metadata
        is still available under ``traj["traj_metadata"]["episode_metadata"]``.

        Returns True to keep the episode, False to discard it.
        Default implementation keeps all episodes.
        """
        import tensorflow as tf

        return tf.constant(value=True)

    def _apply_frame_transforms_to_trajectory(self, traj: dict) -> dict:
        """Apply frame_transforms to each timestep in a trajectory via tf.map_fn.

        Used in trajectory mode to ensure subclass frame_transforms overrides are applied.
        Returns a trajectory dict where each field is stacked across timesteps.
        """
        import tensorflow as tf

        traj_len = tf.shape(traj["actions"])[0]

        def extract_frame(t: int) -> dict:
            """Extract a single frame at timestep t."""
            frame: dict[str, Any] = {}
            for key, value in traj.items():
                if isinstance(value, dict):
                    frame[key] = {k: v[t] for k, v in value.items()}
                else:
                    frame[key] = value[t]
            return frame

        def process_frame(t: int) -> dict:
            """Extract frame at t and apply frame_transforms."""
            frame = extract_frame(t)
            return self.frame_transforms(frame)

        # Use tf.map_fn to apply frame_transforms to each timestep
        # We need to get the output structure from a sample frame
        sample_frame = process_frame(0)

        def map_body(t):
            return process_frame(t)

        # Map over all timesteps
        return tf.map_fn(
            map_body,
            tf.range(traj_len),
            fn_output_signature=tf.nest.map_structure(
                lambda x: tf.TensorSpec(shape=x.shape, dtype=x.dtype), sample_frame
            ),
        )

    def __iter__(self):
        """Iterate over the dataset.

        In training mode: yields batched frame dicts.
        In trajectory mode: yields trajectory dicts (stacked arrays per trajectory).
        """
        yield from self.dataset.as_numpy_iterator()

    def __len__(self) -> int:
        """Return dataset length.

        In trajectory mode: returns number of trajectories (striding already applied at TFDS level).
        In training mode: raises TypeError (infinite streaming has no length).
        """
        if self._return_trajectories:
            return int(self.dataset.cardinality().numpy())
        raise TypeError("Training mode dataset has no length (infinite streaming).")
