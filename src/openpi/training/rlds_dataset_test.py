"""Tests for shared RLDS logic in BaseRldsDataset."""

import sys
import types

import numpy as np
import pytest
import tensorflow as tf

import openpi.training.rlds_dataset as rlds_dataset


def _make_base_dataset(*, discount: float = 0.99, reward_scale: float = 1.0, reward_bias: float = 0.0):
    dataset = rlds_dataset.BaseRldsDataset.__new__(rlds_dataset.BaseRldsDataset)
    dataset._discount = discount  # noqa: SLF001
    dataset._reward_scale = reward_scale  # noqa: SLF001
    dataset._reward_bias = reward_bias  # noqa: SLF001
    dataset._critic_mode = True  # noqa: SLF001
    dataset._action_chunk_size = 3  # noqa: SLF001
    dataset._image_obs_keys = ()  # noqa: SLF001
    return dataset


def test_compute_chunk_rl_fields_boundary_masking():
    dataset = _make_base_dataset(discount=0.99)

    traj_len = 5
    chunk_size = 3
    rewards = tf.constant([0.0, 0.0, 0.0, 0.0, 1.0], dtype=tf.float32)
    terminations = tf.constant([False, False, False, False, True])
    truncations = tf.constant([False] * 5)

    chunk_rewards, chunk_terms, chunk_truncs = dataset._compute_chunk_rl_fields(  # noqa: SLF001
        rewards, terminations, truncations, chunk_size, traj_len
    )

    chunk_rewards = chunk_rewards.numpy()
    chunk_terms = chunk_terms.numpy()
    chunk_truncs = chunk_truncs.numpy()

    np.testing.assert_allclose(chunk_rewards[4], 1.0, rtol=1e-5)
    np.testing.assert_allclose(chunk_rewards[3], 0.99, rtol=1e-5)
    np.testing.assert_allclose(chunk_rewards[2], 0.99**2, rtol=1e-5)
    assert bool(chunk_terms[2])
    assert bool(chunk_terms[3])
    assert bool(chunk_terms[4])
    assert not bool(chunk_terms[1])
    assert not np.any(chunk_truncs)


def test_compute_chunk_rl_fields_rewards_after_termination_are_masked():
    """Rewards occurring after a termination within a chunk should be excluded."""
    dataset = _make_base_dataset(discount=0.99)

    traj_len = 5
    chunk_size = 3
    # Termination at position 2, but non-zero reward at position 3
    rewards = tf.constant([0.0, 0.0, 1.0, 5.0, 0.0], dtype=tf.float32)
    terminations = tf.constant([False, False, True, False, False])
    truncations = tf.constant([False] * 5)

    chunk_rewards, chunk_terms, _ = dataset._compute_chunk_rl_fields(  # noqa: SLF001
        rewards, terminations, truncations, chunk_size, traj_len
    )

    chunk_rewards = chunk_rewards.numpy()
    chunk_terms = chunk_terms.numpy()

    # Position 0: chunk [0,1,2], no done before position 2
    # Should be: 0 + 0 + 0.99^2*1.0 = 0.9801
    np.testing.assert_allclose(chunk_rewards[0], 0.99**2, rtol=1e-5)

    # Position 1: chunk [1,2,3], termination at 2
    # Should be: 0 + 0.99*1.0 + 0 = 0.99 (NOT 0 + 0.99*1.0 + 0.99^2*5.0)
    np.testing.assert_allclose(chunk_rewards[1], 0.99, rtol=1e-5)

    # Position 2: chunk [2,3,4], termination at 2 (first position)
    # Should be: 1.0 + 0 + 0 = 1.0 (done at position 0 of chunk, mask subsequent)
    np.testing.assert_allclose(chunk_rewards[2], 1.0, rtol=1e-5)

    # Verify termination flags still work correctly
    assert bool(chunk_terms[0])  # chunk [0,1,2] contains term at 2
    assert bool(chunk_terms[1])  # chunk [1,2,3] contains term at 2
    assert bool(chunk_terms[2])  # chunk [2,3,4] contains term at 2
    assert not bool(chunk_terms[3])  # chunk [3,4,4] does not contain term at 2
    assert not bool(chunk_terms[4])  # chunk [4,4,4] does not contain term at 2


def test_compute_mc_returns_tf():
    dataset = _make_base_dataset()
    rewards = tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float32)
    dones = tf.constant([False, False, False, True])
    mc_returns = dataset._compute_mc_returns_tf(rewards, dones, 0.99).numpy()  # noqa: SLF001

    np.testing.assert_allclose(mc_returns[3], 1.0, rtol=1e-5)
    np.testing.assert_allclose(mc_returns[2], 0.99, rtol=1e-5)
    np.testing.assert_allclose(mc_returns[1], 0.99**2, rtol=1e-5)
    np.testing.assert_allclose(mc_returns[0], 0.99**3, rtol=1e-5)


def test_apply_rl_fields_shifts_all_observation_keys():
    dataset = _make_base_dataset()
    traj_len = 6
    action_chunk_size = 3

    mapped = {
        "actions": tf.cast(tf.range(traj_len), tf.float32),
        "observation": {
            "state": tf.cast(tf.range(traj_len), tf.float32),
            "image": tf.cast(tf.range(traj_len) + 10, tf.float32),
            "wrist_image": tf.cast(tf.range(traj_len) + 20, tf.float32),
        },
        "prompt": "task",
    }
    raw = {
        "reward": tf.constant([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=tf.float32),
        "is_terminal": tf.constant([False, False, False, True, False, False]),
        "is_truncated": tf.constant([False] * traj_len),
    }

    result = dataset._apply_rl_fields(raw, mapped, action_chunk_size)  # noqa: SLF001
    next_indices = tf.minimum(tf.range(traj_len) + action_chunk_size, traj_len - 1).numpy()

    assert set(result["next_observation"].keys()) == {"state", "image", "wrist_image"}
    np.testing.assert_array_equal(result["next_observation"]["state"].numpy(), next_indices)
    np.testing.assert_array_equal(result["next_observation"]["image"].numpy(), next_indices + 10)
    np.testing.assert_array_equal(result["next_observation"]["wrist_image"].numpy(), next_indices + 20)
    np.testing.assert_array_equal(result["next_actions_raw"].numpy(), next_indices)


def test_apply_rl_fields_derives_counterfactual_next_actions():
    dataset = _make_base_dataset()
    traj_len = 6
    action_chunk_size = 3
    counterfactual_actions = tf.reshape(
        tf.cast(tf.range(traj_len * 2 * 3), tf.float32),
        [traj_len, 2, 3],
    )

    mapped = {
        "actions": tf.cast(tf.range(traj_len), tf.float32),
        "observation": {
            "state": tf.cast(tf.range(traj_len), tf.float32),
        },
        "counterfactual_actions": counterfactual_actions,
        "prompt": "task",
    }
    raw = {
        "reward": tf.constant([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=tf.float32),
        "is_terminal": tf.constant([False, False, False, True, False, False]),
        "is_truncated": tf.constant([False] * traj_len),
    }

    result = dataset._apply_rl_fields(raw, mapped, action_chunk_size)  # noqa: SLF001
    next_indices = tf.minimum(tf.range(traj_len) + action_chunk_size, traj_len - 1)

    tf.debugging.assert_equal(result["counterfactual_actions"], counterfactual_actions)
    tf.debugging.assert_equal(
        result["counterfactual_next_actions"],
        tf.gather(counterfactual_actions, next_indices),
    )


def test_chunk_actions_chunks_next_actions_raw():
    dataset = _make_base_dataset()
    traj = {
        "actions": tf.cast(tf.range(10), tf.float32),
        "next_actions_raw": tf.cast(tf.range(10) + 100, tf.float32),
    }

    chunked = dataset._chunk_actions(traj, action_chunk_size=3)  # noqa: SLF001

    assert "next_actions_raw" not in chunked
    np.testing.assert_array_equal(chunked["actions"][0].numpy(), [0, 1, 2])
    np.testing.assert_array_equal(chunked["next_actions"][0].numpy(), [100, 101, 102])
    np.testing.assert_array_equal(chunked["actions"][8].numpy(), [8, 9, 9])
    np.testing.assert_array_equal(chunked["next_actions"][8].numpy(), [108, 109, 109])


def test_default_done_flags_use_rlds_keys():
    dataset = _make_base_dataset()
    raw = {
        "is_terminal": tf.constant([False, False, True, False]),
        "is_truncated": tf.constant([False, True, False, False]),
    }
    rewards = tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)

    termination, truncation = dataset._get_per_step_done_flags(raw, mapped_traj={}, per_step_rewards=rewards)  # noqa: SLF001
    np.testing.assert_array_equal(termination.numpy(), [False, False, True, False])
    np.testing.assert_array_equal(truncation.numpy(), [False, True, False, False])


def test_default_done_flags_fallback_to_is_last():
    dataset = _make_base_dataset()
    raw = {
        "is_terminal": tf.constant([False, True, False, False]),
        "is_last": tf.constant([False, False, False, True]),
    }
    rewards = tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)

    termination, truncation = dataset._get_per_step_done_flags(raw, mapped_traj={}, per_step_rewards=rewards)  # noqa: SLF001
    np.testing.assert_array_equal(termination.numpy(), [False, True, False, False])
    np.testing.assert_array_equal(truncation.numpy(), [False, False, False, True])


def test_prepare_trajectory_fails_fast_when_done_flags_missing():
    dataset = _make_base_dataset()
    raw = {
        "action": tf.reshape(tf.cast(tf.range(6), tf.float32), [3, 2]),
        "observation": {"state": tf.cast(tf.range(3), tf.float32)},
        "language_instruction": "task",
        "reward": tf.constant([0.0, 0.0, 1.0], dtype=tf.float32),
    }
    dataset_cfg = rlds_dataset.RLDSDataset(name="dummy", version="1.0.0", weight=1.0)

    with pytest.raises(ValueError, match="Could not derive per-step termination flags"):
        dataset._prepare_trajectory(raw, dataset_cfg)  # noqa: SLF001


def test_init_skips_shuffle_when_disabled(monkeypatch: pytest.MonkeyPatch):
    class FakeDataset:
        def __init__(self):
            self.shuffle_calls = []

        def repeat(self):
            return self

        def traj_map(self, fn, num_parallel_calls):
            del fn, num_parallel_calls
            return self

        def flatten(self, num_parallel_calls):
            del num_parallel_calls
            return self

        def frame_map(self, fn, num_parallel_calls):
            del fn, num_parallel_calls
            return self

        def filter(self, fn):
            del fn
            return self

        def shuffle(self, buffer_size):
            self.shuffle_calls.append(buffer_size)
            return self

        def batch(self, batch_size):
            del batch_size
            return self

        def with_ram_budget(self, ram_budget):
            del ram_budget
            return self

    class FakeDLataset:
        sampled_dataset: FakeDataset | None = None

        @staticmethod
        def from_rlds(builder, split, shuffle, num_parallel_reads, read_config_kwargs=None):
            del builder, split, shuffle, num_parallel_reads, read_config_kwargs
            return FakeDataset()

        @classmethod
        def sample_from_datasets(cls, datasets, weights):
            del datasets, weights
            cls.sampled_dataset = FakeDataset()
            return cls.sampled_dataset

    fake_tf = types.SimpleNamespace(config=types.SimpleNamespace(set_visible_devices=lambda *args, **kwargs: None))
    fake_jax = types.SimpleNamespace(process_count=lambda: 1, process_index=lambda: 0)
    fake_tfds = types.SimpleNamespace(
        builder=lambda name, data_dir, version: (name, data_dir, version),
        split_for_jax_process=lambda split, process_index, process_count: split,
    )
    fake_dlimp = types.SimpleNamespace(DLataset=FakeDLataset)

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "tensorflow_datasets", fake_tfds)
    monkeypatch.setitem(sys.modules, "dlimp", fake_dlimp)

    datasets = [rlds_dataset.RLDSDataset(name="dummy", version="1.0.0", weight=1.0)]

    rlds_dataset.BaseRldsDataset(
        data_dir="/tmp",
        batch_size=8,
        datasets=datasets,
        shuffle=False,
        shuffle_buffer_size=123,
    )
    assert FakeDLataset.sampled_dataset is not None
    assert FakeDLataset.sampled_dataset.shuffle_calls == []

    rlds_dataset.BaseRldsDataset(
        data_dir="/tmp",
        batch_size=8,
        datasets=datasets,
        shuffle=True,
        shuffle_buffer_size=123,
    )
    assert FakeDLataset.sampled_dataset is not None
    assert FakeDLataset.sampled_dataset.shuffle_calls == [123]
