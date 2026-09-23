"""RLDS-based data loader for RoboCOIN.

RoboCOIN dataset structure:
    observation/state: 14D float32 (joint positions and gripper)
    observation/image/cam_0: base camera JPEG
    observation/image/cam_1: left wrist camera JPEG
    observation/image/cam_2: right wrist camera JPEG
    action: 14D float32 (absolute joint actions)
    eef_sim_pose_state: 14D EEF pose (optional, for use_eef=True)
    eef_sim_pose_action: 14D EEF actions (optional)
    subtask_1..subtask_5: subtask text strings
    steps_to_subtask_end: [5] int32, steps until each subtask completes
    first_null_index: int32, number of valid subtasks (0-5)

Key features:
    - Multi-subtask episodes: up to 5 subtasks per episode
    - Per-frame subtask sampling during training
    - Next-state indexing controlled by td_n with fps-aware offsets
    - Optional EEF action representation (14D) with joint-angle state
"""

from collections.abc import Sequence
import logging
from typing import Any, Literal

import openpi.training.rlds_dataset as rlds_dataset

SubtaskPromptMode = Literal[
    "subtask_only",
    "all_subtasks",
    "all_subtasks_predict_current_subtask",
    "task_description_predict_current_subtask",
    "task_description",
]

ALL_SUBTASKS_HANG_PROMPT = (
    "Grasp the hanger. Lift the hanger off the rod. Pass hanger from right to left arm. "
    "Hook one side of the shirt onto the hanger. Hook the other side of the shirt onto the hanger. "
    "Place the hanger on the rod."
)

TASK_DESCRIPTION_HANG_PROMPT = "Place the shirt on the hanger and hang it from the rod."


class RoboCoinRldsDataset(rlds_dataset.BaseRldsDataset):
    """RoboCOIN-specific RLDS dataset loader.

    Handles RoboCOIN-specific trajectory key mapping and frame-level subtask selection.
    Image decoding is handled by BaseRldsDataset.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[rlds_dataset.RLDSDataset],
        *,
        split: str = "train",
        shuffle: bool = True,
        shuffle_seed: int = 86,
        action_chunk_size: int = 25,
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
        mask_50fps: bool = False,
        mask_boundary_actions: bool = True,
        variable_horizon: bool = False,
        lower_action_horizon: int = 1,
        use_chunk_wise_delta: bool = False,
        state_dim: int = 14,
        subtask_prompt_mode: SubtaskPromptMode = "subtask_only",
        image_size: tuple[int, int] | None = None,
        include_images: bool = True,
        decode_images: bool = True,
        return_trajectories: bool = False,
        max_trajectories: int | None = None,
        max_num_demos: int | None = None,
        counterfactual_action_store_dir: str | None = None,
        counterfactual_action_dim_offset: int = 0,
    ):
        if td_n is not None and td_n % 5 != 0:
            raise ValueError(f"td_n must be a multiple of 5, got {td_n}")
        if action_chunk_size % 5 != 0:
            raise ValueError(f"action_chunk_size must be a multiple of 5, got {action_chunk_size}")
        if state_dim not in (14, 16):
            raise ValueError(f"state_dim must be 14 or 16, got {state_dim}")
        if state_dim == 16 and use_eef:
            raise ValueError("state_dim=16 requires use_eef=False (zero-padding only applies to raw 14D state).")
        if filter_n is not None and filter_n % 5 != 0:
            raise ValueError(f"filter_n must be a multiple of 5, got {filter_n}")
        if variable_horizon and td_n != action_chunk_size:
            raise ValueError(
                "variable_horizon=True requires td_n == action_chunk_size, "
                f"got td_n={td_n}, action_chunk_size={action_chunk_size}"
            )
        self._split = split
        self._use_eef = use_eef
        self._td_n = td_n
        self._filter_n = filter_n
        self._mask_50fps = mask_50fps
        self._mask_boundary_actions = mask_boundary_actions
        self._variable_horizon = variable_horizon
        self._lower_action_horizon = lower_action_horizon
        self._state_dim = state_dim
        self._state_dim_checked = False
        self._subtask_prompt_mode = subtask_prompt_mode
        self._counterfactual_action_dim_offset = counterfactual_action_dim_offset
        logging.info(
            f"RoboCoinRldsDataset: critic_mode={critic_mode}, discount={discount}, "
            f"reward_scale={reward_scale}, reward_bias={reward_bias}, use_eef={use_eef}, "
            f"td_n={td_n}, filter_n={filter_n}, mask_50fps={mask_50fps}, "
            f"mask_boundary_actions={mask_boundary_actions}, "
            f"variable_horizon={variable_horizon}, "
            f"lower_action_horizon={lower_action_horizon}, "
            f"state_dim={state_dim}, "
            f"subtask_prompt_mode={subtask_prompt_mode}, "
            f"counterfactual_action_dim_offset={counterfactual_action_dim_offset}"
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
        """Extract trajectory fields used by the frame-level transform."""
        import tensorflow as tf

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
            actions = self._construct_eef_repr(raw_action, eef_action)
            state = self._construct_eef_state(raw_state, eef_state)
        else:
            actions = tf.cast(traj["action"], tf.float32)
            state = tf.cast(traj["observation/state"], tf.float32)
        traj_len = tf.shape(state)[0]
        fps = traj["traj_metadata"]["episode_metadata"]["fps"]
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
            "subtask_2": traj["subtask_2"],
            "subtask_3": traj["subtask_3"],
            "subtask_4": traj["subtask_4"],
            "subtask_5": traj["subtask_5"],
            # task_description has no subtasks: count down to the EPISODE end instead, so
            # mc_return / termination / next_action_mask measure distance to task
            # completion, matching the prompt. Mirrors Hdf5RldsDataset. The per-subtask
            # field is [T, 5]; the episode countdown is broadcast across all five slots so
            # frame_transforms' sampled_idx gather is unchanged (every slot is equally valid).
            "steps_to_subtask_end": (
                tf.repeat(
                    tf.cast(traj["_len"] - 1 - traj["_frame_index"], tf.int32)[:, None], 5, axis = 1,
                )
                if self._subtask_prompt_mode == "task_description"
                else traj["steps_to_subtask_end"]
            ),
            # Under task_description every frame is a valid sample regardless of subtask
            # annotation coverage: an episode with first_null_index=0 (no annotated
            # subtasks) would otherwise be masked out entirely by include_subtask, and one
            # with first_null_index=1 confines sampled_idx to slot 0. All five slots carry the
            # same episode countdown, so declaring all five valid is exact, not a fudge.
            "first_null_index": (
                tf.fill(tf.shape(traj["first_null_index"]), tf.cast(5, traj["first_null_index"].dtype))
                if self._subtask_prompt_mode == "task_description"
                else traj["first_null_index"]
            ),
            "fps": fps,
            "repo_id": repo_id,
            "task_description": task_description,
            "embodiment": embodiment,
        }
        if "has_subtask_annotations" in traj["traj_metadata"]["episode_metadata"]:
            result["has_subtask_annotations"] = traj["traj_metadata"]["episode_metadata"]["has_subtask_annotations"]
        metadata_keys = ("index", "episode_index", "_frame_index", "_traj_index")
        if dataset_cfg.name != "robocoin":
            metadata_keys = metadata_keys + ("repo_index",)
        del dataset_cfg

        for key in metadata_keys:
            if key in traj:
                result[key] = traj[key]

        return result

    @staticmethod
    def _construct_eef_repr(data, eef_data):
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
        return RoboCoinRldsDataset._construct_eef_repr(state, eef_state)

    @staticmethod
    def _extract_embodiment(repo_id):
        """Extract the embodiment name from a RoboCOIN repo identifier."""
        import tensorflow as tf

        repo_name = tf.strings.split(repo_id, sep = "/")[-1]
        parts = tf.strings.split(repo_name, sep = "_")
        return tf.strings.reduce_join(parts[:2], separator = "_")

    def _compute_next_indices(self, traj_len, fps):
        import tensorflow as tf

        frame_indices = tf.range(traj_len, dtype = tf.int32)
        if self._td_n is None:
            next_offset = 1
        else:
            next_offset = tf.where(
                tf.equal(fps, 30),
                3 * self._td_n // 5,
                self._td_n,
            )
        return tf.minimum(frame_indices + next_offset, traj_len - 1)

    def _prepare_trajectory(self, raw_traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Attach next-step fields and subtask-conditioned action masks before flattening."""
        import tensorflow as tf

        mapped_traj = self.trajectory_transforms(raw_traj, dataset_cfg)
        if "counterfactual_actions" in raw_traj:
            counterfactual_actions = raw_traj["counterfactual_actions"]
            if self._counterfactual_action_dim_offset > 0:
                action_dim = mapped_traj["actions"].shape[-1]
                if action_dim is None:
                    raise ValueError("Expected static action dimension for RoboCOIN actions.")
                counterfactual_actions = counterfactual_actions[
                    ...,
                    self._counterfactual_action_dim_offset : self._counterfactual_action_dim_offset + action_dim,
                ]
            mapped_traj["counterfactual_actions"] = counterfactual_actions
        for key in ("_ca_episode_index",):
            if key in raw_traj:
                mapped_traj[key] = raw_traj[key]

        traj_len = tf.shape(mapped_traj["actions"])[0]
        fps = tf.cast(mapped_traj["fps"][0], tf.int32)

        offsets = tf.range(self._action_chunk_size, dtype = tf.int32)
        steps = tf.cast(mapped_traj["steps_to_subtask_end"], tf.int32)
        base_action_mask = offsets[None, None, :] <= steps[:, :, None]
        mapped_traj["action_mask"] = base_action_mask

        if self._td_n is None:
            next_indices = self._compute_next_indices(traj_len, fps)
        elif self._variable_horizon:
            sampled_k_cap = tf.where(
                tf.equal(fps, 30),
                tf.constant(3 * self._action_chunk_size // 5, dtype = tf.int32),
                tf.constant(self._action_chunk_size, dtype = tf.int32),
            )
            lower_k = tf.maximum(
                tf.constant(1, dtype = tf.int32),
                tf.where(
                    tf.equal(fps, 30),
                    tf.constant(3 * self._lower_action_horizon // 5, dtype = tf.int32),
                    tf.constant(self._lower_action_horizon, dtype = tf.int32),
                ),
            )
            if self._split == "train":
                sampled_k_native = tf.random.uniform(
                    tf.shape(steps),
                    minval = lower_k,
                    maxval = sampled_k_cap + 1,
                    dtype = tf.int32,
                )
            else:
                sampled_k_native = tf.broadcast_to(sampled_k_cap, tf.shape(steps))
            # With boundary masking, cap the sampled horizon at the number of valid
            # actions until subtask end; otherwise let it cross the subtask boundary.
            if self._mask_boundary_actions:
                k_native = tf.minimum(
                    sampled_k_native, tf.reduce_sum(tf.cast(base_action_mask, tf.int32), axis = -1)
                )
            else:
                k_native = sampled_k_native
            next_indices = tf.minimum(tf.range(traj_len, dtype = tf.int32)[:, None] + k_native, traj_len - 1)
            mapped_traj["variable_k_native"] = k_native
            mapped_traj["action_mask"] = offsets[None, None, :] < k_native[:, :, None]
            if self._mask_boundary_actions:
                mapped_traj["next_action_mask"] = mapped_traj["action_mask"] & (
                    offsets[None, None, :] <= (steps - k_native)[:, :, None]
                )
            else:
                mapped_traj["next_action_mask"] = mapped_traj["action_mask"]
        else:
            next_indices = self._compute_next_indices(traj_len, fps)
            td_n_native = tf.where(
                tf.equal(fps, 30),
                3 * self._td_n // 5,
                self._td_n,
            )
            mapped_traj["next_action_mask"] = offsets[None, None, :] <= (steps - td_n_native)[:, :, None]

        mapped_traj["next_observation"] = {
            key: tf.gather(value, next_indices) for key, value in mapped_traj["observation"].items()
        }

        if self._variable_horizon:
            next_action_indices = tf.minimum(
                tf.range(traj_len, dtype = tf.int32)[:, None, None]
                + mapped_traj["variable_k_native"][:, :, None]
                + offsets[None, None, :],
                traj_len - 1,
            )
        else:
            next_action_indices = tf.minimum(next_indices[:, None] + offsets[None, :], traj_len - 1)
        mapped_traj["next_actions"] = tf.gather(mapped_traj["actions"], next_action_indices)

        if "counterfactual_actions" in mapped_traj:
            mapped_traj["counterfactual_next_actions"] = tf.gather(mapped_traj["counterfactual_actions"], next_indices)

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
        """Decode images, select a subtask, and produce the final per-frame batch keys."""
        import tensorflow as tf

        first_null_index = tf.cast(frame["first_null_index"], tf.int32)
        steps_all = tf.cast(frame["steps_to_subtask_end"], tf.int32)

        if self._return_trajectories:
            sampled_idx = tf.constant(0, dtype = tf.int32)
        else:
            safe_upper = tf.maximum(first_null_index, 1)
            sampled_idx = tf.random.uniform([], minval = 0, maxval = safe_upper, dtype = tf.int32)

        subtask_texts = tf.stack(
            [
                frame["subtask_1"],
                frame["subtask_2"],
                frame["subtask_3"],
                frame["subtask_4"],
                frame["subtask_5"],
            ]
        )
        stripped_subtask_texts = tf.strings.regex_replace(subtask_texts, r"\.\s*$", "")
        lower_subtask_texts = tf.strings.lower(stripped_subtask_texts)
        valid_subtasks = tf.range(5) < first_null_index
        include_subtasks = (
            valid_subtasks
            & tf.not_equal(lower_subtask_texts, b"static")
            & tf.not_equal(lower_subtask_texts, b"abnormal")
        )

        masked_steps = tf.where(include_subtasks, steps_all, tf.int32.max)
        min_idx = tf.argmin(masked_steps, output_type = tf.int32)

        if self._critic_mode:
            # When counterfactual actions are cached, the policy generated them
            # using the min-length valid subtask's mask. Use the same index for
            # action_mask / next_action_mask so ReplaceMaskedActions and the
            # network's attention mask are consistent with the cached actions.
            if self._counterfactual_action_store_dir is not None and self._mask_boundary_actions:
                sampled_idx = min_idx
            frame["prompt"] = subtask_texts[sampled_idx]
        else:
            selected_texts = tf.boolean_mask(stripped_subtask_texts, include_subtasks)
            frame["prompt"] = tf.strings.reduce_join(selected_texts, separator = ", ")
            sampled_idx = min_idx

        if self._variable_horizon:
            frame["next_observation"] = {
                key: value[sampled_idx] for key, value in frame["next_observation"].items()
            }
            frame["next_actions"] = frame["next_actions"][sampled_idx]
            if "counterfactual_next_actions" in frame:
                frame["counterfactual_next_actions"] = frame["counterfactual_next_actions"][sampled_idx]
        else:
            frame["next_actions"] = tf.cast(frame["next_actions"], tf.float32)

        frame = super().frame_transforms(frame)

        images, image_masks = self._restructure_images(frame)
        raw_state = tf.cast(frame["observation"]["state"], tf.float32)

        if self._use_eef:
            if not self._state_dim_checked:
                tf.debugging.assert_equal(
                    tf.shape(raw_state)[-1], 14,
                    message = "use_eef=True requires 14D state inside RoboCoinRldsDataset",
                )
                self._state_dim_checked = True
            frame["state"] = raw_state
        elif self._state_dim == 14:
            if not self._state_dim_checked:
                tf.debugging.assert_equal(
                    tf.shape(raw_state)[-1], 14,
                    message = "state_dim=14 but raw state is not 14D",
                )
                self._state_dim_checked = True
            frame["state"] = raw_state
        elif self._state_dim == 16:
            if not self._state_dim_checked:
                tf.debugging.assert_equal(
                    tf.shape(raw_state)[-1], 14,
                    message = "state_dim=16 expects a 14D raw state to zero-pad in RoboCoinRldsDataset",
                )
                self._state_dim_checked = True
            frame["state"] = tf.concat(
                [raw_state[:6], [0.0], raw_state[6:13], [0.0], raw_state[13:]], axis = 0,
            )

        del frame["observation"]

        frame["actions"] = tf.cast(frame["actions"], tf.float32)

        if images:
            frame["image"] = images
            frame["image_mask"] = image_masks

        raw_next_state = tf.cast(frame["next_observation"]["state"], tf.float32)
        if self._use_eef:
            tf.debugging.assert_equal(
                tf.shape(raw_next_state)[-1], 14,
                message = "use_eef=True requires 14D next_state inside RoboCoinRldsDataset",
            )
            frame["next_state"] = raw_next_state
        elif self._state_dim == 14:
            frame["next_state"] = raw_next_state
        elif self._state_dim == 16:
            frame["next_state"] = tf.concat(
                [raw_next_state[:6], [0.0], raw_next_state[6:13], [0.0], raw_next_state[13:]], axis = 0,
            )

        next_images, next_image_masks = self._restructure_images(frame, prefix = "next_")
        if next_images:
            frame["next_image"] = next_images
            frame["next_image_mask"] = next_image_masks
        del frame["next_observation"]

        selected_steps = steps_all[sampled_idx]
        frame["include_subtask"] = include_subtasks[sampled_idx]
        frame["sampled_index"] = sampled_idx

        full_action_horizon = tf.shape(frame["action_mask"])[1]
        if self._variable_horizon or self._mask_boundary_actions:
            frame["action_mask"] = frame["action_mask"][sampled_idx]
            frame["next_action_mask"] = frame["next_action_mask"][sampled_idx]
        else:
            frame["action_mask"] = tf.ones([full_action_horizon], dtype = tf.bool)
            frame["next_action_mask"] = tf.ones([full_action_horizon], dtype = tf.bool)

        if self._subtask_prompt_mode == "all_subtasks":
            frame["prompt"] = tf.constant(ALL_SUBTASKS_HANG_PROMPT)
            frame["subtask_text"] = tf.constant(b"")
        elif self._subtask_prompt_mode == "all_subtasks_predict_current_subtask":
            frame["prompt"] = tf.constant(ALL_SUBTASKS_HANG_PROMPT)
            frame["subtask_text"] = subtask_texts[sampled_idx]
        elif self._subtask_prompt_mode == "task_description_predict_current_subtask":
            frame["prompt"] = frame["task_description"]
            frame["subtask_text"] = subtask_texts[sampled_idx]
        elif self._subtask_prompt_mode == "task_description":
            frame["prompt"] = frame["task_description"]
            frame["subtask_text"] = tf.constant(b"")

        selected_steps_f = tf.cast(selected_steps, tf.float32)
        frame["steps_to_subtask_end"] = selected_steps

        fps = tf.cast(frame["fps"], tf.int32)
        exponent_per_step = tf.where(
            tf.equal(fps, 30),
            tf.constant(5.0, dtype = tf.float32),
            tf.constant(3.0, dtype = tf.float32),
        )
        frame["mc_return"] = tf.pow(self._discount, exponent_per_step * selected_steps_f)

        td_n = self._td_n if self._td_n is not None else 0
        if self._variable_horizon:
            k_native = tf.cast(frame["variable_k_native"][sampled_idx], tf.int32)
            frame["td_discount"] = tf.pow(self._discount, exponent_per_step * tf.cast(k_native, tf.float32))
        else:
            frame["td_discount"] = tf.pow(self._discount, 3.0 * tf.cast(td_n, tf.float32))

        if self._td_n is not None:
            if self._variable_horizon:
                td_n_native = tf.cast(frame["variable_k_native"][sampled_idx], tf.int32)
            else:
                td_n_native = tf.where(
                    tf.equal(fps, 30),
                    3 * td_n // 5,
                    td_n,
                )
            termination = selected_steps < td_n_native
            td_reward = tf.pow(self._discount, exponent_per_step * selected_steps_f)
            frame["termination"] = termination
            frame["reward"] = tf.where(termination, td_reward, 0.0)
        else:
            frame["termination"] = tf.equal(selected_steps, 0)
            frame["reward"] = tf.cast(frame["termination"], tf.float32)

        frame["truncation"] = tf.constant(False)

        is_30fps = tf.equal(fps, 30)
        action_horizon = tf.shape(frame["action_mask"])[-1]
        valid_30fps_actions = 3 * action_horizon // 5
        fps_mask_30 = tf.sequence_mask(valid_30fps_actions, action_horizon)
        frame["action_mask"] = tf.where(is_30fps, tf.logical_and(frame["action_mask"], fps_mask_30), frame["action_mask"])
        frame["next_action_mask"] = tf.where(
            is_30fps,
            tf.logical_and(frame["next_action_mask"], fps_mask_30),
            frame["next_action_mask"],
        )
        if self._variable_horizon:
            frame["variable_k_native"] = tf.cast(frame["variable_k_native"][sampled_idx], tf.int32)

        return frame

    def _apply_frame_transforms_to_trajectory(self, traj: dict) -> dict:
        """Apply frame transforms to a trajectory and then apply the same frame filter."""
        import tensorflow as tf

        traj = super()._apply_frame_transforms_to_trajectory(traj)

        # Mirror the per-frame predicates in `frame_filter` so trajectory mode applies the
        # same filtering as flat-frame mode (include_subtask, mask_50fps, filter_n).
        mask = tf.cast(traj["include_subtask"], tf.bool)
        if self._mask_50fps:
            mask = tf.logical_and(mask, tf.equal(tf.cast(traj["fps"], tf.int32), 30))
        if self._filter_n is not None:
            fps = tf.cast(traj["fps"], tf.int32)
            filter_n_native = tf.where(
                tf.equal(fps, 30),
                3 * self._filter_n // 5,
                self._filter_n,
            )
            mask = tf.logical_and(mask, traj["steps_to_subtask_end"] >= filter_n_native)
        if "has_subtask_annotations" in traj and self._subtask_prompt_mode != "task_description":
            mask = tf.logical_and(mask, tf.cast(traj["has_subtask_annotations"], tf.bool))
        return tf.nest.map_structure(lambda x: tf.boolean_mask(x, mask), traj)

    def frame_filter(self, frame: dict) -> bool:
        """Filter out frames with invalid subtasks or insufficient horizon."""
        import tensorflow as tf

        keep = tf.constant(True)
        if self._mask_50fps:
            keep = tf.logical_and(keep, tf.equal(tf.cast(frame["fps"], tf.int32), 30))
        keep = tf.logical_and(keep, frame["include_subtask"])

        if self._filter_n is not None:
            fps = tf.cast(frame["fps"], tf.int32)
            filter_n_native = tf.where(
                tf.equal(fps, 30),
                3 * self._filter_n // 5,
                self._filter_n,
            )
            keep = tf.logical_and(keep, frame["steps_to_subtask_end"] >= filter_n_native)

        if "has_subtask_annotations" in frame and self._subtask_prompt_mode != "task_description":
            keep = tf.logical_and(keep, tf.cast(frame["has_subtask_annotations"], tf.bool))

        return keep
