"""Tests for RoboCoinRldsDataset."""

import io

import numpy as np
import PIL.Image
import tensorflow as tf

import openpi.training.rlds_dataset as rlds_dataset
import openpi.training.robocoin_rlds_dataset as robocoin_rlds_dataset

_ROBOCOIN_DATASET_CFG = rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0)


def _create_valid_jpeg_bytes() -> bytes:
    img = PIL.Image.new("RGB", (8, 8), color = (128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format = "JPEG")
    return buffer.getvalue()


VALID_JPEG_BYTES = _create_valid_jpeg_bytes()


def _make_robocoin_dataset_for_unit_tests(
    *,
    use_eef: bool = False,
    critic_mode: bool = False,
    td_n: int | None = None,
    filter_n: int | None = None,
    mask_50fps: bool = False,
    use_chunk_wise_delta: bool = False,
    state_dim: int = 14,
):
    dataset = robocoin_rlds_dataset.RoboCoinRldsDataset.__new__(robocoin_rlds_dataset.RoboCoinRldsDataset)
    # Mirror the attributes __init__ would set without touching a TFDS builder.
    vars(dataset).update(
        {
            "_discount": 0.99,
            "_reward_scale": 1.0,
            "_reward_bias": 0.0,
            "_use_eef": use_eef,
            "_critic_mode": critic_mode,
            "_td_n": td_n,
            "_filter_n": filter_n,
            "_mask_50fps": mask_50fps,
            "_mask_boundary_actions": True,
            "_variable_horizon": False,
            "_lower_action_horizon": 1,
            "_use_chunk_wise_delta": use_chunk_wise_delta,
            "_state_dim": state_dim,
            "_state_dim_checked": False,
            "_subtask_prompt_mode": "subtask_only",
            "_image_obs_keys": ("cam_0", "cam_1", "cam_2"),
            "_image_size": None,
            "_decode_images": True,
            "_action_chunk_size": 10,
            "_return_trajectories": False,
            "_max_trajectories": None,
            "_include_images": True,
            "_counterfactual_action_store_dir": None,
            "_counterfactual_action_dim_offset": 0,
        }
    )
    return dataset


def _make_mock_trajectory(traj_len: int = 12) -> dict:
    return {
        "action": tf.reshape(tf.range(traj_len * 14, dtype = tf.float32), [traj_len, 14]),
        "eef_sim_pose_action": tf.reshape(tf.range(traj_len * 12, dtype = tf.float32), [traj_len, 12]) + 1000.0,
        "eef_sim_pose_state": tf.reshape(tf.range(traj_len * 12, dtype = tf.float32), [traj_len, 12]) + 2000.0,
        "observation/state": tf.reshape(tf.range(traj_len * 14, dtype = tf.float32), [traj_len, 14]) + 500.0,
        "observation/image/cam_0": tf.constant([VALID_JPEG_BYTES] * traj_len),
        "observation/image/cam_1": tf.constant([VALID_JPEG_BYTES] * traj_len),
        "observation/image/cam_2": tf.constant([VALID_JPEG_BYTES] * traj_len),
        "subtask_1": tf.constant([b"pick up the cup."] * traj_len),
        "subtask_2": tf.constant([b"place on table"] * traj_len),
        "subtask_3": tf.constant([b"static"] * traj_len),
        "subtask_4": tf.constant([b"abnormal"] * traj_len),
        "subtask_5": tf.constant([b"null"] * traj_len),
        "steps_to_subtask_end": tf.constant([[9, 4, 3, 2, 0]] * traj_len, dtype = tf.int32),
        "first_null_index": tf.constant([4] * traj_len, dtype = tf.int32),
        "episode_index": tf.constant([7] * traj_len, dtype = tf.int32),
        "_frame_index": tf.range(traj_len, dtype = tf.int32),
        "_traj_index": tf.constant([0] * traj_len, dtype = tf.int32),
        # dlimp broadcasts episode metadata to one entry per step.
        "traj_metadata": {
            "episode_metadata": {
                "fps": tf.constant([30.0] * traj_len, dtype = tf.float32),
                "repo_id": tf.constant([b"RoboCOIN/Split_aloha_plate_storage"] * traj_len),
                "task_description": tf.constant([b"store the plate"] * traj_len),
            },
        },
    }


def _make_frame_for_transforms(
    *,
    first_null_index: int = 4,
    steps_to_subtask_end: list[int] | tuple[int, ...] = (9, 4, 3, 2, 0),
    fps: int = 30,
) -> dict:
    return {
        "observation": {
            "state": tf.constant(np.arange(14), dtype = tf.float32),
            "cam_0": tf.constant(VALID_JPEG_BYTES),
            "cam_1": tf.constant(VALID_JPEG_BYTES),
            "cam_2": tf.constant(VALID_JPEG_BYTES),
        },
        "next_observation": {
            "state": tf.constant(np.arange(14) + 100, dtype = tf.float32),
            "cam_0": tf.constant(VALID_JPEG_BYTES),
            "cam_1": tf.constant(VALID_JPEG_BYTES),
            "cam_2": tf.constant(VALID_JPEG_BYTES),
        },
        "actions": tf.constant(np.arange(140).reshape(10, 14), dtype = tf.float32),
        "next_actions": tf.constant(np.arange(140, 280).reshape(10, 14), dtype = tf.float32),
        "action_mask": tf.constant(
            [
                [True] * 10,
                [True] * 5 + [False] * 5,
                [True] * 4 + [False] * 6,
                [True] * 3 + [False] * 7,
                [False] * 10,
            ],
            dtype = tf.bool,
        ),
        "next_action_mask": tf.constant(
            [
                [True] * 10,
                [True] * 5 + [False] * 5,
                [True] * 4 + [False] * 6,
                [True] * 3 + [False] * 7,
                [False] * 10,
            ],
            dtype = tf.bool,
        ),
        "subtask_1": tf.constant(b"pick up the cup."),
        "subtask_2": tf.constant(b"place on table"),
        "subtask_3": tf.constant(b"static"),
        "subtask_4": tf.constant(b"abnormal"),
        "subtask_5": tf.constant(b"null"),
        "steps_to_subtask_end": tf.constant(list(steps_to_subtask_end), dtype = tf.int32),
        "first_null_index": tf.constant(first_null_index, dtype = tf.int32),
        "fps": tf.constant(fps, dtype = tf.int32),
        "repo_id": tf.constant(b"RoboCOIN/Split_aloha_plate_storage"),
    }


class TestEefRepresentation:
    def test_construct_eef_state_shape_and_values(self):
        state = tf.constant([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype = tf.float32)
        eef_state = tf.constant([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]], dtype = tf.float32)

        result = robocoin_rlds_dataset.RoboCoinRldsDataset._construct_eef_state(state, eef_state)  # noqa: SLF001

        assert result.shape == (1, 14)
        np.testing.assert_array_equal(
            result.numpy()[0],
            [100, 101, 102, 103, 104, 105, 6, 106, 107, 108, 109, 110, 111, 13],
        )


class TestTrajectoryTransforms:
    def test_trajectory_transforms_uses_absolute_actions_and_joint_state(self):
        dataset = _make_robocoin_dataset_for_unit_tests(use_eef = False)
        mapped = dataset.trajectory_transforms(_make_mock_trajectory(), dataset_cfg = _ROBOCOIN_DATASET_CFG)

        assert "observation" in mapped
        np.testing.assert_array_equal(mapped["actions"].numpy()[0], np.arange(14, dtype = np.float32))
        np.testing.assert_array_equal(mapped["observation"]["state"].numpy()[0], np.arange(14, dtype = np.float32) + 500.0)
        assert mapped["repo_id"].numpy()[0] == b"RoboCOIN/Split_aloha_plate_storage"

    def test_trajectory_transforms_uses_eef_actions_and_eef_state(self):
        dataset = _make_robocoin_dataset_for_unit_tests(use_eef = True)
        mapped = dataset.trajectory_transforms(_make_mock_trajectory(), dataset_cfg = _ROBOCOIN_DATASET_CFG)

        np.testing.assert_array_equal(
            mapped["actions"].numpy()[0],
            [1000, 1001, 1002, 1003, 1004, 1005, 6, 1006, 1007, 1008, 1009, 1010, 1011, 13],
        )
        np.testing.assert_array_equal(
            mapped["observation"]["state"].numpy()[0],
            [2000, 2001, 2002, 2003, 2004, 2005, 506, 2006, 2007, 2008, 2009, 2010, 2011, 513],
        )

    def test_frame_transforms_keeps_eef_state_14d_when_state_dim_is_16(self):
        dataset = _make_robocoin_dataset_for_unit_tests(use_eef = True, state_dim = 16)
        result = dataset.frame_transforms(_make_frame_for_transforms())

        assert result["state"].shape == (14,)
        assert result["next_state"].shape == (14,)

    def test_prepare_trajectory_uses_td_n_with_fps_aware_offset(self):
        dataset = _make_robocoin_dataset_for_unit_tests(critic_mode = True, td_n = 10)
        traj = _make_mock_trajectory(traj_len = 20)
        prepared = dataset._prepare_trajectory(traj, dataset_cfg = _ROBOCOIN_DATASET_CFG)  # noqa: SLF001

        expected_index = 6
        np.testing.assert_array_equal(
            prepared["next_observation"]["state"].numpy()[0],
            traj["observation/state"].numpy()[expected_index],
        )
        np.testing.assert_array_equal(prepared["next_actions"].numpy()[0, 0], traj["action"].numpy()[expected_index])

    def test_prepare_trajectory_builds_subtask_action_masks(self):
        dataset = _make_robocoin_dataset_for_unit_tests(critic_mode = True)
        prepared = dataset._prepare_trajectory(_make_mock_trajectory(), dataset_cfg = _ROBOCOIN_DATASET_CFG)  # noqa: SLF001

        assert prepared["action_mask"].shape == (12, 5, 10)
        np.testing.assert_array_equal(
            prepared["action_mask"].numpy()[0, 1],
            [True, True, True, True, True, False, False, False, False, False],
        )

class TestFrameTransforms:
    def test_critic_mode_selects_sampled_subtask_and_computes_td_fields(self):
        dataset = _make_robocoin_dataset_for_unit_tests(critic_mode = True, td_n = 10)
        tf.random.set_seed(86)

        result = dataset.frame_transforms(_make_frame_for_transforms())

        assert result["prompt"].numpy() in {b"pick up the cup.", b"place on table", b"static", b"abnormal"}
        assert result["state"].shape == (14,)
        assert result["next_state"].shape == (14,)
        assert result["actions"].shape == (10, 14)
        assert result["next_actions"].shape == (10, 14)
        assert "include_subtask" in result
        np.testing.assert_allclose(result["td_discount"].numpy(), np.float32(0.99 ** 30), rtol = 1e-6)
        np.testing.assert_array_equal(result["action_mask"].numpy()[6:], [False, False, False, False])

    def test_policy_mode_concatenates_valid_non_special_subtasks(self):
        dataset = _make_robocoin_dataset_for_unit_tests(critic_mode = False)
        result = dataset.frame_transforms(_make_frame_for_transforms())

        assert result["prompt"].numpy() == b"pick up the cup, place on table"
        assert int(result["sampled_index"].numpy()) == 1
        assert int(result["steps_to_subtask_end"].numpy()) == 4

    def test_policy_mode_keeps_actions_absolute_under_chunk_wise_delta(self):
        # The chunk-wise delta is applied by the DeltaActions data transform that the config
        # installs, so the dataset itself must hand out absolute actions.
        dataset = _make_robocoin_dataset_for_unit_tests(critic_mode = False, use_chunk_wise_delta = True)
        result = dataset.frame_transforms(_make_frame_for_transforms(fps = 50))

        np.testing.assert_array_equal(result["actions"].numpy()[0], np.arange(14, dtype = np.float32))
        np.testing.assert_array_equal(result["next_actions"].numpy()[0], np.arange(140, 154, dtype = np.float32))

    def test_include_subtask_excludes_static_and_abnormal_selected_subtasks(self):
        dataset = _make_robocoin_dataset_for_unit_tests(critic_mode = False)
        frame = _make_frame_for_transforms(first_null_index = 4, steps_to_subtask_end = (9, 4, 1, 0, 0))
        result = dataset.frame_transforms(frame)

        assert bool(result["include_subtask"].numpy())

        bad_frame = _make_frame_for_transforms(first_null_index = 1, steps_to_subtask_end = (1, 0, 0, 0, 0))
        bad_frame["subtask_1"] = tf.constant(b"static")
        bad_result = dataset.frame_transforms(bad_frame)
        assert not bool(bad_result["include_subtask"].numpy())

class TestFiltering:
    def test_frame_filter_uses_fps_aware_filter_n(self):
        dataset = _make_robocoin_dataset_for_unit_tests(filter_n = 10)

        keep_frame = {"steps_to_subtask_end": tf.constant(6, dtype = tf.int32), "fps": tf.constant(30, dtype = tf.int32), "first_null_index": tf.constant(4, dtype = tf.int32), "include_subtask": tf.constant(True)}
        drop_frame = {"steps_to_subtask_end": tf.constant(5, dtype = tf.int32), "fps": tf.constant(30, dtype = tf.int32), "first_null_index": tf.constant(4, dtype = tf.int32), "include_subtask": tf.constant(True)}

        assert bool(dataset.frame_filter(keep_frame).numpy())
        assert not bool(dataset.frame_filter(drop_frame).numpy())

    def test_frame_with_zero_first_null_index_is_filtered_out(self):
        dataset = _make_robocoin_dataset_for_unit_tests()
        frame = dataset.frame_transforms(_make_frame_for_transforms(first_null_index = 0))
        assert not bool(frame["include_subtask"].numpy())
        assert not bool(dataset.frame_filter(frame).numpy())

    def test_frame_filter_rejects_excluded_subtask(self):
        dataset = _make_robocoin_dataset_for_unit_tests()
        frame = {"steps_to_subtask_end": tf.constant(10, dtype = tf.int32), "fps": tf.constant(30, dtype = tf.int32), "first_null_index": tf.constant(4, dtype = tf.int32), "include_subtask": tf.constant(False)}
        assert not bool(dataset.frame_filter(frame).numpy())

    def test_frame_filter_rejects_50fps_when_mask_50fps_enabled(self):
        dataset = _make_robocoin_dataset_for_unit_tests(mask_50fps = True)
        frame = {"steps_to_subtask_end": tf.constant(10, dtype = tf.int32), "fps": tf.constant(50, dtype = tf.int32), "first_null_index": tf.constant(4, dtype = tf.int32), "include_subtask": tf.constant(True)}
        assert not bool(dataset.frame_filter(frame).numpy())
