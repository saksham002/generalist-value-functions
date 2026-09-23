"""RLDS data loader for LeRobot-built real-world datasets (e.g. ``realworld_xarm_packing``).

Sibling of ``Hdf5RldsDataset`` for datasets exported by the LeRobot RLDS builder.
Unlike the HDF5 schema, here ``observation/state`` and ``action`` are usually already
14D EEF (``pos3 + euler3 + gripper`` per arm, no separate ``eef_sim_pose_*`` fields to
reconstruct); joint-space datasets (e.g. ``lego``) are the exception and do ship 12D
``eef_sim_pose_action`` / ``eef_sim_pose_state``, which ``use_eef=True`` splices with
the joint grippers to rebuild the 14D EEF layout. The subtask annotation is a single
per-step ``subtask`` string, and every demo is an expert/complete trajectory (no
``reward`` field, no ``is_partial`` heuristic, no ``has_subtask_annotations``).

Datasets that do carry partial demos annotate them per SUBTASK, exposed as a per-step
``is_partial`` bool. Setting ``filter_partial=True`` drops each partial subtask's
trailing ``td_n`` (or ``action_chunk_size`` when ``td_n`` is None) steps,
independently of ``critic_mode`` and of ``prompt_mode``, so a truncated subtask never
anchors a reward.

Supports both behavior cloning (``critic_mode=False``) and value-function training
(``critic_mode=True``). In critic mode the next-step gathers, action masks and
sparse-reward / MC-return / TD fields are produced by ``_apply_rl_fields`` (ported
from ``Hdf5RldsDataset`` minus the variable-horizon and is-partial logic).
``subsample=True`` downsamples raw 60 Hz episodes to 30 Hz before field extraction.
"""

from collections.abc import Sequence
import logging
from typing import Any, Literal

import openpi.training.hdf5_rlds_dataset as hdf5_rlds_dataset
import openpi.training.rlds_dataset as rlds_dataset

PromptMode = Literal["subtask", "task_description", "task_description_predict_current_subtask"]


class LeRobotRldsDataset(rlds_dataset.BaseRldsDataset):
    """LeRobot-sourced RLDS dataset loader with single-subtask, expert-only semantics."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[rlds_dataset.RLDSDataset],
        *,
        split: str = "train",
        shuffle: bool = True,
        shuffle_seed: int = 86,
        action_chunk_size: int = 30,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        critic_mode: bool = False,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        use_eef: bool = False,
        td_n: int | None = None,
        filter_n: int | None = None,
        filter_partial: bool = False,
        filter_adversarial: bool = False,
        mask_boundary_actions: bool = True,
        prompt_mode: PromptMode = "subtask",
        subsample: bool = False,
        image_size: tuple[int, int] | None = None,
        include_images: bool = True,
        decode_images: bool = True,
        return_trajectories: bool = False,
        max_trajectories: int | None = None,
        max_num_demos: int | None = None,
        counterfactual_action_store_dir: str | None = None,
        counterfactual_action_dim_offset: int = 0,
    ):
        if action_chunk_size not in (30, 60):
            raise ValueError(f"action_chunk_size must be 30 or 60, got {action_chunk_size}")
        if td_n is not None and td_n % 2 != 0:
            raise ValueError(f"td_n must be a multiple of 2, got {td_n}")
        if filter_n is not None and filter_n % 2 != 0:
            raise ValueError(f"filter_n must be a multiple of 2, got {filter_n}")

        self._split = split
        self._use_eef = use_eef
        self._td_n = td_n
        self._filter_n = filter_n
        self._filter_partial = filter_partial
        self._filter_adversarial = filter_adversarial
        self._mask_boundary_actions = mask_boundary_actions
        self._prompt_mode = prompt_mode
        self._subsample = subsample
        self._counterfactual_action_dim_offset = counterfactual_action_dim_offset
        logging.info(
            f"LeRobotRldsDataset: critic_mode={critic_mode}, use_eef={use_eef}, td_n={td_n}, "
            f"filter_n={filter_n}, filter_partial={filter_partial}, filter_adversarial={filter_adversarial}, "
            f"mask_boundary_actions={mask_boundary_actions}, prompt_mode={prompt_mode}, subsample={subsample}"
        )

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            datasets=datasets,
            split=split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            action_chunk_size=action_chunk_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            critic_mode=critic_mode,
            discount=discount,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            image_obs_keys=("cam_0", "cam_1", "cam_2"),
            image_size=image_size,
            include_images=include_images,
            decode_images=decode_images,
            return_trajectories=return_trajectories,
            max_trajectories=max_trajectories,
            max_num_demos=max_num_demos,
            counterfactual_action_store_dir=counterfactual_action_store_dir,
        )

    def trajectory_transforms(self, traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Extract the per-trajectory fields consumed by ``_apply_rl_fields`` / ``frame_transforms``."""
        import tensorflow as tf

        del dataset_cfg

        if self._subsample:
            traj = self._subsample_trajectory(traj)

        actions = tf.cast(traj["action"], tf.float32)
        state = tf.cast(traj["observation/state"], tf.float32)

        # Most LeRobot datasets are already 14D EEF, so `action` / `observation/state` are
        # used as-is. Joint-space ones (e.g. lego) additionally ship 12D `eef_sim_pose_*`,
        # from which the 14D EEF layout is rebuilt by splicing in the two gripper slots —
        # the same construction Hdf5RldsDataset uses. Keyed on field presence rather than
        # on use_eef alone so that use_eef=True stays correct for both schemas; `traj` is a
        # plain dict at trace time, so this costs nothing at runtime.
        has_eef_pose = "eef_sim_pose_action" in traj or "eef_sim_pose_state" in traj
        if self._use_eef and has_eef_pose:
            if "eef_sim_pose_action" not in traj or "eef_sim_pose_state" not in traj:
                raise ValueError(
                    "use_eef=True needs both eef_sim_pose_action and eef_sim_pose_state; "
                    "found only one, which would pair an EEF action with a joint state."
                )
            actions = hdf5_rlds_dataset.Hdf5RldsDataset.construct_eef_repr(
                actions, tf.cast(traj["eef_sim_pose_action"], tf.float32)
            )
            state = hdf5_rlds_dataset.Hdf5RldsDataset.construct_eef_repr(
                state, tf.cast(traj["eef_sim_pose_state"], tf.float32)
            )

        tf.debugging.assert_equal(
            tf.shape(state)[-1], 14, message="LeRobotRldsDataset requires 14D observation/state",
        )
        tf.debugging.assert_equal(
            tf.shape(actions)[-1], 14, message="LeRobotRldsDataset requires 14D action",
        )

        fps = traj["traj_metadata"]["episode_metadata"]["fps"]
        # Invariant: fps=30 must be reached via subsample=True; raw source data is 60 Hz.
        if not self._subsample:
            tf.debugging.assert_equal(
                tf.cast(fps[0], tf.int32), tf.constant(60, dtype=tf.int32),
                message="LeRobotRldsDataset requires fps=60 source data when subsample=False.",
            )

        observation = {"state": state}
        if self._include_images:
            observation["cam_0"] = traj["observation/image/cam_0"]
            observation["cam_1"] = traj["observation/image/cam_1"]
            observation["cam_2"] = traj["observation/image/cam_2"]

        # task_description has no subtasks: count down to episode end instead.
        if self._prompt_mode == "task_description":
            steps_to_subtask_end = tf.cast(traj["_len"] - 1 - traj["_frame_index"], tf.int32)
        else:
            steps_to_subtask_end = tf.cast(traj["steps_to_subtask_end"], tf.int32)

        result = {
            "actions": actions,
            "observation": observation,
            "subtask": traj["subtask"],
            "task_description": traj["traj_metadata"]["episode_metadata"]["task_description"],
            "steps_to_subtask_end": steps_to_subtask_end,
            "fps": fps,
            # repo_id lives only in episode_metadata; forward it (and per-step repo_index
            # below) so cache_val_episodes can key validation trajectories by repo.
            # Mirrors Hdf5RldsDataset.
            "repo_id": traj["traj_metadata"]["episode_metadata"]["repo_id"],
        }

        # Partiality is per-SUBTASK here, not per-episode. Read the per-STEP `is_partial`
        # field rather than gathering episode_metadata/subtask_is_partial by subtask_index:
        # dlimp broadcasts a variable-length episode_metadata array to a FLATTENED
        # [T * num_subtasks] tile (not [num_subtasks]), which is still rank 1 and so slips
        # past a rank check, and _subsample_trajectory's per-step stride-2 slice then
        # reorders it into sip[(2j) % num_subtasks] — wrong for every subtask but the
        # first. `is_partial` is a genuine [T] field carrying the same values, so it
        # subsamples correctly and needs no gather. Unlike Hdf5RldsDataset there is no
        # heuristic to infer it.
        if "is_partial" in traj:
            result["is_partial"] = tf.cast(traj["is_partial"], tf.bool)
        if self._filter_partial:
            if "is_partial" not in traj:
                raise ValueError(
                    "filter_partial=True requires a per-step is_partial field, "
                    "which this dataset does not provide."
                )
            # task_description overwrote steps_to_subtask_end with the countdown to the
            # EPISODE end above, so carry the subtask countdown separately for this filter:
            # partiality is a subtask-level property and its tail must be measured against
            # the subtask. Other prompt modes already have it under steps_to_subtask_end.
            if self._prompt_mode == "task_description":
                result["steps_to_subtask_end_for_partial"] = tf.cast(traj["steps_to_subtask_end"], tf.int32)

        # Per-step passthrough used by validation caching (cache_val_episodes sorts
        # frames by _frame_index and keys trajectories by repo_index). Guarded so
        # absent keys are skipped. Mirrors Hdf5RldsDataset's passthrough set.
        for key in ("index", "episode_index", "_frame_index", "_traj_index", "repo_index", "frame_index", "_len"):
            if key in traj:
                result[key] = traj[key]

        # Subsample case only: counterfactual_actions / _ca_episode_index were already
        # subsampled by _subsample_trajectory and must be forwarded with the matching
        # (T//2) leading dim (the non-subsample case reads them in _prepare_trajectory).
        if self._subsample:
            if "counterfactual_actions" in traj:
                result["counterfactual_actions"] = traj["counterfactual_actions"]
            if "_ca_episode_index" in traj:
                result["_ca_episode_index"] = traj["_ca_episode_index"]

        return result

    def _subsample_trajectory(self, traj: dict) -> dict:
        """Subsample a 60 Hz trajectory to 30 Hz before any field extraction.

        Action-bearing leaves are sliced ``[1::2]`` (kept action sits at the half-step
        between the surrounding 30 Hz states); all other leaves ``[0::2]``. Each
        per-step counterfactual chunk is itself a 60 Hz rollout, so it is subsampled on
        both the trajectory axis (``[0::2]``, tracking state) and the action_horizon
        axis (``[1::2]`` + zero-pad). fps is overwritten with 30.
        """
        import tensorflow as tf

        fps = traj["traj_metadata"]["episode_metadata"]["fps"]
        tf.debugging.assert_equal(
            tf.cast(fps, tf.int32), tf.constant(60, dtype=tf.int32),
            message="subsample=True requires the underlying dataset to be 60 Hz.",
        )

        traj_len = tf.shape(traj["action"])[0]
        m = traj_len // 2
        end = 2 * m

        def _walk(value, key: str):
            if isinstance(value, dict):
                return {k: _walk(v, k) for k, v in value.items()}
            if key == "counterfactual_actions":
                trajectory_sliced = value[0:end:2]
                horizon_sliced = trajectory_sliced[:, :, 1::2, :]
                pad_amount = tf.shape(trajectory_sliced)[2] - tf.shape(horizon_sliced)[2]
                return tf.pad(horizon_sliced, [[0, 0], [0, 0], [0, pad_amount], [0, 0]])
            start = 1 if "action" in key else 0
            return value[start:end:2]

        out = {k: _walk(v, k) for k, v in traj.items()}

        for halve_key in ("_frame_index", "index", "steps_to_subtask_end", "_len"):
            if halve_key in out:
                out[halve_key] = out[halve_key] // 2

        ep_meta = out["traj_metadata"]["episode_metadata"]
        ep_meta["fps"] = tf.fill(tf.shape(ep_meta["fps"]), tf.cast(30, ep_meta["fps"].dtype))

        return out

    def _partial_tail_window(self, fps):
        """Native-step window dropped from the end of every partial subtask.

        Mirrors Hdf5RldsDataset's is_partial tail filter, but keyed on the subtask
        rather than the episode, and applied regardless of critic_mode: with td_n set
        the window is td_n (the bootstrap span), otherwise it is the action horizon.
        Both are in 60 Hz units, so fps=30 halves them.
        """
        import tensorflow as tf

        span = self._td_n if self._td_n is not None else self._action_chunk_size
        return tf.where(tf.equal(fps, 30), span // 2, span)

    def _compute_next_indices(self, traj_len, fps):
        import tensorflow as tf

        # td_n is in 60 Hz units. At fps=30 (post-subsample) each native step covers
        # two 60 Hz steps, so native = td_n // 2.
        frame_indices = tf.range(traj_len, dtype=tf.int32)
        if self._td_n is None:
            next_offset = 1
        else:
            next_offset = tf.where(tf.equal(fps, 30), self._td_n // 2, self._td_n)
        return tf.minimum(frame_indices + next_offset, traj_len - 1)

    def _prepare_trajectory(self, raw_traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Map fields, then run ``_apply_rl_fields`` only in critic mode (BC stays lean)."""
        mapped_traj = self.trajectory_transforms(raw_traj, dataset_cfg)
        if "actions" not in mapped_traj:
            raise ValueError("trajectory_transforms must produce an 'actions' key for action chunking.")

        # Non-subsample case: read counterfactual actions straight from the raw join
        # (the subsample case already forwarded them via trajectory_transforms).
        if "counterfactual_actions" in raw_traj and not self._subsample:
            counterfactual_actions = raw_traj["counterfactual_actions"]
            if self._counterfactual_action_dim_offset > 0:
                action_dim = mapped_traj["actions"].shape[-1]
                if action_dim is None:
                    raise ValueError("Expected static action dimension for LeRobot actions.")
                counterfactual_actions = counterfactual_actions[
                    ...,
                    self._counterfactual_action_dim_offset : self._counterfactual_action_dim_offset + action_dim,
                ]
            mapped_traj["counterfactual_actions"] = counterfactual_actions
        for key in ("_ca_episode_index",):
            if key in raw_traj and not self._subsample:
                mapped_traj[key] = raw_traj[key]

        if self._critic_mode:
            return self._apply_rl_fields(raw_traj, mapped_traj, self._action_chunk_size)
        return mapped_traj

    def _apply_rl_fields(self, raw_traj: dict, mapped_traj: dict, action_chunk_size: int) -> dict:
        """Per-trajectory critic fields: next-step gathers, masks, reward / mc_return / td."""
        import tensorflow as tf

        del raw_traj

        traj_len = tf.shape(mapped_traj["actions"])[0]
        fps = tf.cast(mapped_traj["fps"][0], tf.int32)
        steps = tf.cast(mapped_traj["steps_to_subtask_end"], tf.int32)
        mapped_traj["steps_to_subtask_end"] = steps

        offsets = tf.range(action_chunk_size, dtype=tf.int32)
        base_action_mask = offsets[None, :] <= steps[:, None]

        if self._td_n is None:
            next_indices = self._compute_next_indices(traj_len, fps)
            action_mask = base_action_mask
            next_action_mask = base_action_mask
        else:
            next_indices = self._compute_next_indices(traj_len, fps)
            td_n_native = tf.where(tf.equal(fps, 30), self._td_n // 2, self._td_n)
            action_mask = base_action_mask
            next_action_mask = offsets[None, :] <= (steps - td_n_native)[:, None]

        # Mirror Hdf5RldsDataset: without mask_boundary_actions, treat every action slot
        # as valid (the fps-prefix clamp below still applies at fps=30).
        if not self._mask_boundary_actions:
            action_mask = tf.ones_like(action_mask, dtype=tf.bool)
            next_action_mask = tf.ones_like(next_action_mask, dtype=tf.bool)

        # action_chunk_size is in 60 Hz units; at fps=30 only the first half is valid.
        is_30fps = tf.equal(fps, 30)
        valid_30fps_actions = action_chunk_size // 2
        fps_mask_30 = tf.sequence_mask(valid_30fps_actions, action_chunk_size)
        action_mask = tf.where(is_30fps, tf.logical_and(action_mask, fps_mask_30), action_mask)
        next_action_mask = tf.where(is_30fps, tf.logical_and(next_action_mask, fps_mask_30), next_action_mask)

        mapped_traj["action_mask"] = action_mask
        mapped_traj["next_action_mask"] = next_action_mask

        observation = mapped_traj["observation"]
        if not isinstance(observation, dict):
            raise ValueError("Expected observation to be a dict.")
        mapped_traj["next_observation"] = {
            key: tf.gather(value, next_indices) for key, value in observation.items()
        }
        # Per-step next action; BaseRldsDataset._chunk_actions chunks it into next_actions.
        mapped_traj["next_actions_raw"] = tf.gather(mapped_traj["actions"], next_indices)

        if "counterfactual_actions" in mapped_traj:
            mapped_traj["counterfactual_next_actions"] = tf.gather(
                mapped_traj["counterfactual_actions"], next_indices
            )

        # Per-step RL fields. `exponent_per_step` keeps the {5.0, 2.5} pair (targets
        # assigned on an underlying 150 Hz MDP, i.e. exp_per_step = 150 / fps).
        steps_f = tf.cast(steps, tf.float32)
        exponent_per_step = tf.where(
            tf.equal(fps, 30),
            tf.constant(5.0, dtype=tf.float32),
            tf.constant(2.5, dtype=tf.float32),
        )
        mapped_traj["mc_return"] = tf.pow(self._discount, exponent_per_step * steps_f)
        # `is_partial` is per subtask here, so it is the MC-return validity mask directly:
        # a partial subtask never reached its success state and the countdown-based
        # return above is fiction for it. Datasets without the field have no partial
        # demos (see module docstring), so every return is trustworthy.
        if "is_partial" in mapped_traj:
            mapped_traj["mc_return_mask"] = tf.logical_not(mapped_traj["is_partial"])
        else:
            mapped_traj["mc_return_mask"] = tf.ones_like(steps, dtype=tf.bool)

        td_n = self._td_n if self._td_n is not None else 0
        # 5.0 * td_n/2 = 2.5 * td_n at fps=30 and 2.5 * td_n at fps=60.
        mapped_traj["td_discount"] = tf.fill(
            tf.shape(steps_f),
            tf.pow(self._discount, 2.5 * tf.cast(td_n, tf.float32)),
        )

        if self._td_n is not None:
            td_n_native_per_step = tf.fill(
                tf.shape(steps),
                tf.where(tf.equal(fps, 30), self._td_n // 2, self._td_n),
            )
            termination = steps < td_n_native_per_step
            td_reward = tf.pow(self._discount, exponent_per_step * steps_f)
            mapped_traj["termination"] = termination
            mapped_traj["reward"] = tf.where(termination, td_reward, tf.zeros_like(td_reward))
        else:
            mapped_traj["termination"] = tf.equal(steps, 0)
            mapped_traj["reward"] = tf.cast(mapped_traj["termination"], tf.float32)

        mapped_traj["truncation"] = tf.zeros_like(mapped_traj["termination"], dtype=tf.bool)

        return mapped_traj

    def _restructure_images(self, frame: dict[str, Any], prefix: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
        """Move camera images from observation dicts into the standard image/image_mask format."""
        import tensorflow as tf

        images = {}
        masks = {}
        image_name_map = {
            "cam_0": "base_0_rgb",
            "cam_1": "left_wrist_0_rgb",
            "cam_2": "right_wrist_0_rgb",
        }
        observation_key = "observation" if prefix == "" else "next_observation"
        if observation_key not in frame:
            return images, masks

        for key, value in frame[observation_key].items():
            if key == "state":
                continue
            image_key = image_name_map[key]
            images[image_key] = value
            masks[image_key] = tf.constant(True)
        return images, masks

    def frame_transforms(self, frame: dict) -> dict:
        """Decode images, set the prompt, restructure to standard openpi keys."""
        import tensorflow as tf

        if self._prompt_mode == "task_description":
            frame["prompt"] = frame["task_description"]
        elif self._prompt_mode == "task_description_predict_current_subtask":
            frame["prompt"] = frame["task_description"]
            frame["subtask_text"] = frame["subtask"]
        else:
            frame["prompt"] = frame["subtask"]

        if "next_actions" in frame:
            frame["next_actions"] = tf.cast(frame["next_actions"], tf.float32)

        frame = super().frame_transforms(frame)

        images, image_masks = self._restructure_images(frame)
        frame["state"] = tf.cast(frame["observation"]["state"], tf.float32)
        del frame["observation"]

        frame["actions"] = tf.cast(frame["actions"], tf.float32)
        if images:
            frame["image"] = images
            frame["image_mask"] = image_masks

        # Critic mode: next-step observation/state/images for the TD backup.
        if "next_observation" in frame:
            frame["next_state"] = tf.cast(frame["next_observation"]["state"], tf.float32)
            next_images, next_image_masks = self._restructure_images(frame, prefix="next_")
            if next_images:
                frame["next_image"] = next_images
                frame["next_image_mask"] = next_image_masks
            del frame["next_observation"]

        return frame

    def _apply_frame_transforms_to_trajectory(self, traj: dict) -> dict:
        """Apply frame transforms, then mirror frame_filter on the trajectory."""
        import tensorflow as tf

        traj = super()._apply_frame_transforms_to_trajectory(traj)

        traj_len = tf.shape(traj["actions"])[0]
        mask = tf.ones([traj_len], dtype=tf.bool)
        if self._filter_n is not None:
            fps = tf.cast(traj["fps"], tf.int32)
            filter_n_native = tf.where(tf.equal(fps, 30), self._filter_n // 2, self._filter_n)
            mask = tf.logical_and(mask, traj["steps_to_subtask_end"] >= filter_n_native)
        # Drop the tail of every partial subtask: those steps have no terminal anchor
        # (no successful subtask end to bootstrap toward or imitate).
        if self._filter_partial:
            window = self._partial_tail_window(tf.cast(traj["fps"], tf.int32))
            steps_key = (
                "steps_to_subtask_end_for_partial"
                if self._prompt_mode == "task_description"
                else "steps_to_subtask_end"
            )
            tail = traj[steps_key] < window
            mask = tf.logical_and(mask, tf.logical_not(tf.logical_and(tail, traj["is_partial"])))
        return tf.nest.map_structure(lambda x: tf.boolean_mask(x, mask), traj)

    def trajectory_filter(self, traj: dict) -> bool:
        """Drop whole episodes flagged adversarial when ``filter_adversarial`` is set."""
        import tensorflow as tf

        if not self._filter_adversarial:
            return tf.constant(value=True)

        episode_metadata = traj["traj_metadata"]["episode_metadata"]
        if "is_adversarial" not in episode_metadata:
            raise ValueError(
                "filter_adversarial=True requires episode_metadata/is_adversarial, "
                f"which this dataset does not provide (keys: {sorted(episode_metadata)})."
            )
        # dlimp broadcasts scalar episode_metadata to [T], so take the first entry.
        return tf.logical_not(tf.cast(episode_metadata["is_adversarial"][0], tf.bool))

    def frame_filter(self, frame: dict) -> bool:
        """Drop frames within ``filter_n`` steps of the (sub)task end."""
        import tensorflow as tf

        keep = tf.constant(True)
        if self._filter_n is not None:
            fps = tf.cast(frame["fps"], tf.int32)
            filter_n_native = tf.where(tf.equal(fps, 30), self._filter_n // 2, self._filter_n)
            keep = tf.logical_and(keep, frame["steps_to_subtask_end"] >= filter_n_native)
        # Mirror _apply_frame_transforms_to_trajectory.
        if self._filter_partial:
            window = self._partial_tail_window(tf.cast(frame["fps"], tf.int32))
            steps_key = (
                "steps_to_subtask_end_for_partial"
                if self._prompt_mode == "task_description"
                else "steps_to_subtask_end"
            )
            tail = frame[steps_key] < window
            keep = tf.logical_and(keep, tf.logical_not(tf.logical_and(tail, frame["is_partial"])))
        return keep
