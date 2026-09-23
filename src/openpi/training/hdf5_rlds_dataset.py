"""RLDS data loader for HDF5-sourced real-world datasets (e.g. ``real_hang``).

Leaner sibling of ``RoboCoinRldsDataset``: assumes a single-subtask schema
(``subtask_1`` and ``task_description`` only, no ``subtask_2..5`` /
``first_null_index``), fps in {30, 60}, and ``action_chunk_size`` in {30, 60}.
``subsample=True`` downsamples raw 60 Hz episodes to 30 Hz before any field
extraction; when ``subsample=False`` the source fps is asserted to be 60.

RL-field computation (``mc_return``, ``reward``, ``termination``, ``truncation``,
``td_discount``) plus the per-trajectory next-step gathers
(``next_observation``, ``next_actions``, ``action_mask``, ``next_action_mask``)
are produced once per trajectory inside an override of
``BaseRldsDataset._apply_rl_fields`` rather than smeared across
``_prepare_trajectory`` and ``frame_transforms``.
"""

from collections.abc import Sequence
import logging
from typing import Any, Literal

import openpi.training.rlds_dataset as rlds_dataset

PromptMode = Literal["subtask", "task_description", "task_description_predict_current_subtask"]


class Hdf5RldsDataset(rlds_dataset.BaseRldsDataset):
    """HDF5-sourced RLDS dataset loader with single-subtask semantics."""

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
        critic_mode: bool = True,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        use_eef: bool = False,
        td_n: int | None = None,
        filter_n: int | None = None,
        filter_intervention: bool = False,
        filter_repo_index: tuple[int, ...] | None = None,
        mask_boundary_actions: bool = True,
        variable_horizon: bool = False,
        use_chunk_wise_delta: bool = False,
        state_dim: int = 16,
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
        allow_unannotated_episodes: bool = False,
    ):
        if allow_unannotated_episodes and not return_trajectories:
            raise ValueError("Unannotated episodes are supported only for trajectory evaluation.")
        if action_chunk_size not in (30, 60):
            raise ValueError(f"action_chunk_size must be 30 or 60, got {action_chunk_size}")
        # Source HDF5 trajectories always carry a 14D raw state. The valid output
        # combinations are: (state_dim=16, use_eef=False) -> zero-pad raw 14D to 16D;
        # (state_dim=14, use_eef=True) -> build 14D from EEF + gripper. No other
        # combination is supported.
        if not ((state_dim == 16 and not use_eef) or (state_dim == 14 and use_eef)):
            raise ValueError(
                "Hdf5RldsDataset requires (state_dim=16, use_eef=False) or "
                f"(state_dim=14, use_eef=True); got state_dim={state_dim}, use_eef={use_eef}"
            )
        if td_n is not None and td_n % 2 != 0:
            raise ValueError(f"td_n must be a multiple of 2, got {td_n}")
        if filter_n is not None and filter_n % 2 != 0:
            raise ValueError(f"filter_n must be a multiple of 2, got {filter_n}")
        if variable_horizon and mask_boundary_actions:
            raise ValueError("variable_horizon=True requires mask_boundary_actions=False")
        if variable_horizon and td_n != action_chunk_size:
            raise ValueError(
                "variable_horizon=True requires td_n == action_chunk_size, "
                f"got td_n={td_n}, action_chunk_size={action_chunk_size}"
            )

        self._split = split
        self._use_eef = use_eef
        self._td_n = td_n
        self._filter_n = filter_n
        self._filter_intervention = filter_intervention
        self._filter_repo_index = filter_repo_index
        self._mask_boundary_actions = mask_boundary_actions
        self._variable_horizon = variable_horizon
        self._state_dim = state_dim
        self._state_dim_checked = False
        self._prompt_mode = prompt_mode
        self._counterfactual_action_dim_offset = counterfactual_action_dim_offset
        self._subsample = subsample
        self._allow_unannotated_episodes = allow_unannotated_episodes
        logging.info(
            f"Hdf5RldsDataset: critic_mode={critic_mode}, discount={discount}, "
            f"reward_scale={reward_scale}, reward_bias={reward_bias}, use_eef={use_eef}, "
            f"td_n={td_n}, filter_n={filter_n}, filter_intervention={filter_intervention}, "
            f"filter_repo_index={filter_repo_index}, "
            f"mask_boundary_actions={mask_boundary_actions}, "
            f"variable_horizon={variable_horizon}, "
            f"state_dim={state_dim}, "
            f"prompt_mode={prompt_mode}, "
            f"counterfactual_action_dim_offset={counterfactual_action_dim_offset}, "
            f"subsample={subsample}"
        )

        super().__init__(
            data_dir = data_dir,
            batch_size = batch_size,
            datasets = datasets,
            split = split,
            shuffle = shuffle,
            shuffle_seed = shuffle_seed,
            action_chunk_size = action_chunk_size,
            shuffle_buffer_size = shuffle_buffer_size,
            num_parallel_reads = num_parallel_reads,
            num_parallel_calls = num_parallel_calls,
            critic_mode = critic_mode,
            discount = discount,
            reward_scale = reward_scale,
            reward_bias = reward_bias,
            image_obs_keys = ("cam_0", "cam_1", "cam_2"),
            image_size = image_size,
            include_images = include_images,
            decode_images = decode_images,
            return_trajectories = return_trajectories,
            max_trajectories = max_trajectories,
            max_num_demos = max_num_demos,
            counterfactual_action_store_dir = counterfactual_action_store_dir,
        )

    def trajectory_transforms(self, traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Extract trajectory fields used by ``_apply_rl_fields`` and ``frame_transforms``."""
        import tensorflow as tf


        # Episode-level completion flag written by the RLDS builder: True when the episode
        # ended before the task succeeded, so the trailing subtask never reached its
        # success state. The RL fields below use it to mask the MC-return diagnostics
        # and the TD tail filter (see ``_apply_rl_fields``).
        episode_metadata = traj["traj_metadata"]["episode_metadata"]
        if "is_partial" not in episode_metadata:
            raise KeyError(
                "This RLDS dataset has no episode_metadata/is_partial field. Datasets read by "
                "Hdf5RldsDataset must record whether each episode ended before the task succeeded; "
                "rebuild it with a builder that writes that flag (see the README's dataset section)."
            )
        is_partial_scalar = tf.cast(episode_metadata["is_partial"][0], tf.bool)

        if self._subsample:
            traj = self._subsample_trajectory(traj)

        if self._use_eef:
            eef_action = tf.cast(traj["eef_sim_pose_action"], tf.float32)
            eef_state = tf.cast(traj["eef_sim_pose_state"], tf.float32)
            raw_action = tf.cast(traj["action"], tf.float32)
            raw_state = tf.cast(traj["observation/state"], tf.float32)
            # 14D EEF construction assumes 12D EEF pose (6 xyz+rpy per arm) and raw joint dim 14 or 16.
            tf.debugging.assert_equal(
                tf.shape(eef_state)[-1], 12,
                message = "use_eef=True requires 12D eef_sim_pose_state",
            )
            tf.debugging.assert_equal(
                tf.shape(eef_action)[-1], 12,
                message = "use_eef=True requires 12D eef_sim_pose_action",
            )
            raw_state_dim = tf.shape(raw_state)[-1]
            raw_action_dim = tf.shape(raw_action)[-1]
            tf.debugging.assert_equal(
                tf.logical_or(tf.equal(raw_state_dim, 14), tf.equal(raw_state_dim, 16)),
                True,
                message = "use_eef=True requires raw observation/state of dim 14 or 16",
            )
            tf.debugging.assert_equal(
                tf.logical_or(tf.equal(raw_action_dim, 14), tf.equal(raw_action_dim, 16)),
                True,
                message = "use_eef=True requires raw action of dim 14 or 16",
            )
            actions = self.construct_eef_repr(raw_action, eef_action)
            state = self._construct_eef_state(raw_state, eef_state)
        else:
            actions = tf.cast(traj["action"], tf.float32)
            state = tf.cast(traj["observation/state"], tf.float32)
        traj_len = tf.shape(state)[0]
        fps = traj["traj_metadata"]["episode_metadata"]["fps"]
        # Invariant: fps=30 must be reached via subsample=True; raw source data must be 60 Hz.
        if not self._subsample:
            tf.debugging.assert_equal(
                tf.cast(fps[0], tf.int32), tf.constant(60, dtype = tf.int32),
                message = "Hdf5RldsDataset requires fps=60 source data when subsample=False.",
            )
        repo_id = traj["traj_metadata"]["episode_metadata"]["repo_id"]
        task_description = traj["traj_metadata"]["episode_metadata"]["task_description"]
        embodiment = tf.repeat(self._extract_embodiment(repo_id[0])[None], traj_len)

        observation = {"state": state}
        if self._include_images:
            observation["cam_0"] = traj["observation/image/cam_0"]
            observation["cam_1"] = traj["observation/image/cam_1"]
            observation["cam_2"] = traj["observation/image/cam_2"]

        result = {
            "actions": actions,
            "observation": observation,
            "subtask_1": traj["subtask_1"],
            # task_description has no subtasks: count down to episode end instead.
            "steps_to_subtask_end": (
                tf.cast(traj["_len"] - 1 - traj["_frame_index"], tf.int32)
                if self._prompt_mode == "task_description"
                else traj["steps_to_subtask_end"]
            ),
            "fps": fps,
            "repo_id": repo_id,
            "task_description": task_description,
            "embodiment": embodiment,
            "is_partial": tf.fill([traj_len], is_partial_scalar),
        }
        if self._filter_intervention:
            result["is_intervention"] = traj["is_intervention"]
        if self._prompt_mode != "task_description":
            # Per-frame episode flag so the filters can drop subtask-prompt frames lacking subtask annotations.
            # task_description_predict_current_subtask still needs the per-frame subtask annotation
            # (it predicts it), so the flag is kept for that mode too — mirrors RoboCOIN.
            result["has_subtask_annotations"] = tf.fill(
                [traj_len],
                tf.cast(traj["traj_metadata"]["episode_metadata"]["has_subtask_annotations"][0], tf.bool),
            )
        del dataset_cfg

        # `frame_index` is the original per-episode frame index baked into the raw HDF5
        # data; we carry it through verbatim. `_subsample_trajectory` keeps it on the
        # [0::2] slice and does NOT divide it by 2, so the same physical step retains
        # the same `frame_index` whether or not subsample=True — handy for matching the
        # same sample across subsampled and un-subsampled views.
        for key in ("index", "episode_index", "_frame_index", "_traj_index", "repo_index", "frame_index", "_len"):
            if key in traj:
                result[key] = traj[key]

        # Subsample case only: _subsample_trajectory has produced subsampled
        # counterfactual_actions and _ca_episode_index whose leading dims match
        # the subsampled main fields (T//2). Forward them so _prepare_trajectory
        # doesn't need to re-read the un-subsampled raw versions (which would
        # cause a from_tensor_slices leading-dim mismatch).
        if self._subsample:
            if "counterfactual_actions" in traj:
                result["counterfactual_actions"] = traj["counterfactual_actions"]
            if "_ca_episode_index" in traj:
                result["_ca_episode_index"] = traj["_ca_episode_index"]

        return result

    def _subsample_trajectory(self, traj: dict) -> dict:
        """Subsample a 60 Hz trajectory to 30 Hz before any field extraction.

        - Leaves whose key contains ``"action"`` are sliced ``[1::2]`` so each kept
          action sits at the half-step between the two surrounding 30 Hz states.
        - All other leaves are sliced ``[0::2]``.
        - ``counterfactual_actions`` is treated separately (see below).
        - All leaves are truncated to the same length ``M = traj_len // 2`` so
          downstream ``from_tensor_slices`` sees consistent first-axis sizes for
          odd-length trajectories.
        - ``_frame_index``, ``index`` and ``steps_to_subtask_end`` are then divided
          by 2 to convert from 60 Hz to 30 Hz units.
        - ``traj_metadata.episode_metadata.fps`` is overwritten with 30.

        ``counterfactual_actions`` special case (shape ``(N, k, ah, ad)``):
        each per-step chunk is itself a 60 Hz action rollout, so subsample BOTH
        the trajectory axis and the chunk's action_horizon axis.
        - Axis 0 (trajectory): ``[0::2]`` — the CF chunk at index t corresponds
          to state t, not the half-step action, so it tracks state's offset (not
          regular action's ``[1::2]``).
        - Axis 2 (action_horizon): ``[1::2]`` — same half-step rule the per-step
          ``actions`` field follows, applied within each chunk.
        - Axis 2 right-padded with zeros back to the original action_horizon so
          downstream shape contracts hold. The data-time ``next_action_mask`` at
          fps=30 is ``[True]*(ah//2) + [False]*(ah - ah//2)``, which precisely
          covers the kept-vs-padded boundary; the zero pad is inert at the
          target network forward.
        """
        import tensorflow as tf

        fps = traj["traj_metadata"]["episode_metadata"]["fps"]
        tf.debugging.assert_equal(
            tf.cast(fps, tf.int32), tf.constant(60, dtype = tf.int32),
            message = "subsample=True requires the underlying dataset to be 60 Hz.",
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
                return tf.pad(
                    horizon_sliced, [[0, 0], [0, 0], [0, pad_amount], [0, 0]],
                )
            start = 1 if "action" in key else 0
            return value[start:end:2]

        out = {k: _walk(v, k) for k, v in traj.items()}

        for halve_key in ("_frame_index", "index", "steps_to_subtask_end", "_len"):
            if halve_key in out:
                out[halve_key] = out[halve_key] // 2

        ep_meta = out["traj_metadata"]["episode_metadata"]
        ep_meta["fps"] = tf.fill(tf.shape(ep_meta["fps"]), tf.cast(30, ep_meta["fps"].dtype))

        return out

    @staticmethod
    def construct_eef_repr(data, eef_data):
        """Construct 14D EEF representation from 12D EEF pose and two gripper slots of `data`.

        EEF pose is always laid out as [left_xyz(3), left_rpy(3), right_xyz(3), right_rpy(3)].
        Gripper positions within `data` depend on its dim: left at dim//2 - 1, right at dim - 1,
        matching the 14D (6, 13) and 16D (7, 15) raw-state layouts.
        """
        import tensorflow as tf

        total_dim = tf.shape(data)[-1]
        left_gripper_index = total_dim // 2 - 1
        right_gripper_index = total_dim - 1

        tf.debugging.assert_equal(
            tf.shape(eef_data)[-1], 12,
            message = "_construct_eef_repr expects 12D EEF data (6 xyz+rpy dims per arm)",
        )

        return tf.concat(
            [
                tf.gather(eef_data, tf.range(6), axis = -1),
                tf.gather(data, [left_gripper_index], axis = -1),
                tf.gather(eef_data, tf.range(6, 12), axis = -1),
                tf.gather(data, [right_gripper_index], axis = -1),
            ],
            axis = -1,
        )

    @staticmethod
    def _construct_eef_state(state, eef_state):
        """Compatibility alias for the 14D EEF representation helper."""
        return Hdf5RldsDataset.construct_eef_repr(state, eef_state)

    @staticmethod
    def _extract_embodiment(repo_id):
        """Extract the embodiment name from a real-world repo identifier."""
        import tensorflow as tf

        repo_name = tf.strings.split(repo_id, sep = "/")[-1]
        parts = tf.strings.split(repo_name, sep = "_")
        return tf.strings.reduce_join(parts[:2], separator = "_")

    def _compute_next_indices(self, traj_len, fps):
        import tensorflow as tf

        # td_n is in 60 Hz units. At fps=60 it maps to td_n native steps; at fps=30
        # (post-subsample) each native step covers two 60 Hz steps, so native = td_n // 2.
        frame_indices = tf.range(traj_len, dtype = tf.int32)
        if self._td_n is None:
            next_offset = 1
        else:
            next_offset = tf.where(tf.equal(fps, 30), self._td_n // 2, self._td_n)
        return tf.minimum(frame_indices + next_offset, traj_len - 1)

    def _prepare_trajectory(self, raw_traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Override base so that ``_apply_rl_fields`` runs even when critic_mode=False.

        The next-step gathers (``next_observation``, ``next_actions``, ``action_mask``,
        ``next_action_mask``) are consumed by ``frame_transforms`` regardless of
        critic_mode, so they must always be set. The base only invokes
        ``_apply_rl_fields`` when critic_mode is True.
        """
        mapped_traj = self.trajectory_transforms(raw_traj, dataset_cfg)
        if "actions" not in mapped_traj:
            raise ValueError("trajectory_transforms must produce an 'actions' key for action chunking.")

        # When self._subsample is True, counterfactual_actions and _ca_episode_index
        # are already subsampled by _subsample_trajectory and passed through
        # trajectory_transforms' result. In the non-subsample case we still need to
        # read them from raw_traj here.
        if "counterfactual_actions" in raw_traj and not self._subsample:
            counterfactual_actions = raw_traj["counterfactual_actions"]
            if self._counterfactual_action_dim_offset > 0:
                action_dim = mapped_traj["actions"].shape[-1]
                if action_dim is None:
                    raise ValueError("Expected static action dimension for HDF5 actions.")
                counterfactual_actions = counterfactual_actions[
                    ...,
                    self._counterfactual_action_dim_offset : self._counterfactual_action_dim_offset + action_dim,
                ]
            mapped_traj["counterfactual_actions"] = counterfactual_actions
        for key in ("_ca_episode_index",):
            if key in raw_traj and not self._subsample:
                mapped_traj[key] = raw_traj[key]

        return self._apply_rl_fields(raw_traj, mapped_traj, self._action_chunk_size)

    def _apply_rl_fields(self, raw_traj: dict, mapped_traj: dict, action_chunk_size: int) -> dict:
        """Per-trajectory critic-mode fields plus the next-step gathers.

        Computes ``action_mask`` / ``next_action_mask`` / ``next_observation`` /
        ``next_actions`` (always), and ``mc_return`` / ``reward`` / ``termination``
        / ``truncation`` / ``td_discount`` (always, but only meaningful when
        critic_mode is True downstream).
        """
        import tensorflow as tf
        del raw_traj

        traj_len = tf.shape(mapped_traj["actions"])[0]
        fps = tf.cast(mapped_traj["fps"][0], tf.int32)

        # `steps_to_subtask_end` is stored as [T, 5] for schema compatibility with
        # the multi-subtask format; HDF5 episodes only populate subtask_1, so we
        # collapse to the per-step scalar via column 0.
        steps_raw = tf.cast(mapped_traj["steps_to_subtask_end"], tf.int32)
        if len(steps_raw.shape) == 2:
            steps = steps_raw[:, 0]
        else:
            steps = steps_raw
        mapped_traj["steps_to_subtask_end"] = steps

        offsets = tf.range(action_chunk_size, dtype = tf.int32)
        base_action_mask = offsets[None, :] <= steps[:, None]

        variable_k_native = None
        if self._td_n is None:
            next_indices = self._compute_next_indices(traj_len, fps)
            action_mask = base_action_mask
            next_action_mask = base_action_mask
        elif self._variable_horizon:
            # action_chunk_size is also in 60 Hz units; native cap = chunk_size at fps=60,
            # chunk_size // 2 at fps=30.
            sampled_k_cap = tf.where(
                tf.equal(fps, 30),
                tf.constant(action_chunk_size // 2, dtype = tf.int32),
                tf.constant(action_chunk_size, dtype = tf.int32),
            )
            if self._split == "train":
                sampled_k_native = tf.random.uniform(
                    tf.shape(steps),
                    minval = 1,
                    maxval = sampled_k_cap + 1,
                    dtype = tf.int32,
                )
            else:
                sampled_k_native = tf.broadcast_to(sampled_k_cap, tf.shape(steps))
            k_native = tf.minimum(sampled_k_native, tf.reduce_sum(tf.cast(base_action_mask, tf.int32), axis = -1))
            next_indices = tf.minimum(tf.range(traj_len, dtype = tf.int32) + k_native, traj_len - 1)
            action_mask = offsets[None, :] < k_native[:, None]
            next_action_mask = action_mask & (offsets[None, :] <= (steps - k_native)[:, None])
            variable_k_native = k_native
        else:
            next_indices = self._compute_next_indices(traj_len, fps)
            td_n_native = tf.where(tf.equal(fps, 30), self._td_n // 2, self._td_n)
            action_mask = base_action_mask
            next_action_mask = offsets[None, :] <= (steps - td_n_native)[:, None]

        # Mirror RoboCoinRldsDataset.frame_transforms: when neither variable_horizon nor
        # mask_boundary_actions is set, ignore the sse-derived masks and treat every action
        # slot as valid (the fps-prefix clamp below still applies for fps=30).
        if not (self._variable_horizon or self._mask_boundary_actions):
            action_mask = tf.ones_like(action_mask, dtype = tf.bool)
            next_action_mask = tf.ones_like(next_action_mask, dtype = tf.bool)

        # action_chunk_size is in 60 Hz units; at fps=30 (post-subsample) only the first
        # half of that span is valid (each native step covers two 60 Hz steps).
        is_30fps = tf.equal(fps, 30)
        valid_30fps_actions = action_chunk_size // 2
        fps_mask_30 = tf.sequence_mask(valid_30fps_actions, action_chunk_size)
        action_mask = tf.where(is_30fps, tf.logical_and(action_mask, fps_mask_30), action_mask)
        next_action_mask = tf.where(
            is_30fps,
            tf.logical_and(next_action_mask, fps_mask_30),
            next_action_mask,
        )

        mapped_traj["action_mask"] = action_mask
        mapped_traj["next_action_mask"] = next_action_mask

        observation = mapped_traj["observation"]
        if not isinstance(observation, dict):
            raise ValueError("Expected observation to be a dict.")
        mapped_traj["next_observation"] = {
            key: tf.gather(value, next_indices) for key, value in observation.items()
        }
        next_action_indices = tf.minimum(next_indices[:, None] + offsets[None, :], traj_len - 1)
        mapped_traj["next_actions"] = tf.gather(mapped_traj["actions"], next_action_indices)

        if "counterfactual_actions" in mapped_traj:
            mapped_traj["counterfactual_next_actions"] = tf.gather(
                mapped_traj["counterfactual_actions"], next_indices
            )

        # Per-step RL fields (mc_return / reward / termination / truncation / td_discount).
        # `exponent_per_step` keeps the {5.0, 2.5} pair to match pre-training (robocoin
        # joint fps=30/50 runs assigned targets on an underlying MDP running at 150 Hz,
        # i.e. exp_per_step = 150 / fps).
        steps_f = tf.cast(steps, tf.float32)
        exponent_per_step = tf.where(
            tf.equal(fps, 30),
            tf.constant(5.0, dtype = tf.float32),
            tf.constant(2.5, dtype = tf.float32),
        )
        mapped_traj["mc_return"] = tf.pow(self._discount, exponent_per_step * steps_f)
        # `is_partial` is per episode here: the final subtask never reached its success
        # state, so the countdown-based return above is fiction for every frame whose
        # subtask runs to the episode end. Earlier subtasks did complete (the next one
        # started), so their returns stand. Consumers mask MC diagnostics with this.
        frame_indices = tf.range(traj_len, dtype = tf.int32)
        subtask_ends_at_episode_end = (frame_indices + steps) >= (traj_len - 1)
        mapped_traj["mc_return_mask"] = tf.logical_not(
            tf.logical_and(mapped_traj["is_partial"], subtask_ends_at_episode_end)
        )

        td_n = self._td_n if self._td_n is not None else 0
        if self._variable_horizon:
            k_native_f = tf.cast(variable_k_native, tf.float32)
            mapped_traj["td_discount"] = tf.pow(self._discount, exponent_per_step * k_native_f)
        else:
            # td_n is in 60 Hz units; `exp_per_step * td_n_native` collapses to 2.5*td_n at
            # both fps (5.0 * td_n/2 = 2.5 * td_n at fps=30 and 2.5 * td_n at fps=60).
            mapped_traj["td_discount"] = tf.fill(
                tf.shape(steps_f),
                tf.pow(self._discount, 2.5 * tf.cast(td_n, tf.float32)),
            )

        if self._td_n is not None:
            if self._variable_horizon:
                td_n_native_per_step = variable_k_native
            else:
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

        mapped_traj["truncation"] = tf.zeros_like(mapped_traj["termination"], dtype = tf.bool)

        if variable_k_native is not None:
            mapped_traj["variable_k_native"] = variable_k_native

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
        """Decode images, set prompt, restructure to standard openpi keys."""
        import tensorflow as tf

        if self._prompt_mode == "task_description":
            frame["prompt"] = frame["task_description"]
        elif self._prompt_mode == "task_description_predict_current_subtask":
            frame["prompt"] = frame["task_description"]
            frame["subtask_text"] = frame["subtask_1"]
        else:
            frame["prompt"] = frame["subtask_1"]

        frame["next_actions"] = tf.cast(frame["next_actions"], tf.float32)

        frame = super().frame_transforms(frame)

        images, image_masks = self._restructure_images(frame)
        raw_state = tf.cast(frame["observation"]["state"], tf.float32)

        if self._use_eef:
            # use_eef=True ⇒ state_dim=14: trajectory_transforms already built the 14D EEF state.
            if not self._state_dim_checked:
                tf.debugging.assert_equal(
                    tf.shape(raw_state)[-1], 14,
                    message = "use_eef=True requires 14D state inside Hdf5RldsDataset",
                )
                self._state_dim_checked = True
            frame["state"] = raw_state
        else:
            # use_eef=False ⇒ state_dim=16: raw HDF5 state is already 16D, passthrough.
            if not self._state_dim_checked:
                tf.debugging.assert_equal(
                    tf.shape(raw_state)[-1], 16,
                    message = "use_eef=False requires 16D raw state inside Hdf5RldsDataset",
                )
                self._state_dim_checked = True
            frame["state"] = raw_state

        del frame["observation"]

        frame["actions"] = tf.cast(frame["actions"], tf.float32)

        if images:
            frame["image"] = images
            frame["image_mask"] = image_masks

        raw_next_state = tf.cast(frame["next_observation"]["state"], tf.float32)
        if self._use_eef:
            tf.debugging.assert_equal(
                tf.shape(raw_next_state)[-1], 14,
                message = "use_eef=True requires 14D next_state inside Hdf5RldsDataset",
            )
        else:
            tf.debugging.assert_equal(
                tf.shape(raw_next_state)[-1], 16,
                message = "use_eef=False requires 16D raw next_state inside Hdf5RldsDataset",
            )
        frame["next_state"] = raw_next_state

        next_images, next_image_masks = self._restructure_images(frame, prefix = "next_")
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
        mask = tf.ones([traj_len], dtype = tf.bool)
        if self._filter_n is not None:
            fps = tf.cast(traj["fps"], tf.int32)
            filter_n_native = tf.where(tf.equal(fps, 30), self._filter_n // 2, self._filter_n)
            mask = tf.logical_and(mask, traj["steps_to_subtask_end"] >= filter_n_native)
        # Drop last td_n frames only when the trajectory is partial (no terminal
        # reward to anchor the bootstrap target). `_len` and `_frame_index` are
        # in raw 60 Hz units regardless of subsample, matching td_n's units.
        if self._td_n is not None:
            fps0 = tf.cast(traj["fps"][0], tf.int32)
            td_n_native = tf.where(tf.equal(fps0, 30), self._td_n // 2, self._td_n)
            tail = (traj["_len"] - int(self._critic_mode == True) - traj["_frame_index"]) < td_n_native
            mask = tf.logical_and(mask, tf.logical_not(tf.logical_and(tail, traj["is_partial"])))
        if self._filter_intervention:
            mask = tf.logical_and(mask, tf.cast(traj["is_intervention"], tf.bool))
        if self._filter_repo_index is not None:
            allowed = tf.constant(self._filter_repo_index, dtype = tf.int32)
            repo_index = tf.cast(traj["repo_index"], tf.int32)
            in_allowed = tf.reduce_any(tf.equal(repo_index[:, None], allowed[None, :]), axis = -1)
            mask = tf.logical_and(mask, in_allowed)
        if self._prompt_mode != "task_description" and not self._allow_unannotated_episodes:
            # Mirror frame_filter.
            mask = tf.logical_and(mask, tf.cast(traj["has_subtask_annotations"], tf.bool))
        return tf.nest.map_structure(lambda x: tf.boolean_mask(x, mask), traj)

    def frame_filter(self, frame: dict) -> bool:
        """Filter frames by horizon (``filter_n``) and intervention (``filter_intervention``)."""
        import tensorflow as tf

        keep = tf.constant(True)

        if self._filter_n is not None:
            fps = tf.cast(frame["fps"], tf.int32)
            filter_n_native = tf.where(tf.equal(fps, 30), self._filter_n // 2, self._filter_n)
            keep = tf.logical_and(keep, frame["steps_to_subtask_end"] >= filter_n_native)

        if self._td_n is not None:
            fps = tf.cast(frame["fps"], tf.int32)
            td_n_native = tf.where(tf.equal(fps, 30), self._td_n // 2, self._td_n)
            tail = (frame["_len"] - int(self._critic_mode == True) - frame["_frame_index"]) < td_n_native
            keep = tf.logical_and(keep, tf.logical_not(tf.logical_and(tail, frame["is_partial"])))

        if self._filter_intervention:
            keep = tf.logical_and(keep, tf.cast(frame["is_intervention"], tf.bool))

        if self._filter_repo_index is not None:
            allowed = tf.constant(self._filter_repo_index, dtype = tf.int32)
            keep = tf.logical_and(
                keep, tf.reduce_any(tf.equal(tf.cast(frame["repo_index"], tf.int32), allowed))
            )

        if self._prompt_mode != "task_description" and not self._allow_unannotated_episodes:
            keep = tf.logical_and(keep, tf.cast(frame["has_subtask_annotations"], tf.bool))

        return keep
