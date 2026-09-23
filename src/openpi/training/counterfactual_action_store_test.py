"""Tests for counterfactual_action_store module."""

import tempfile

import numpy as np
import pytest

from openpi.training import counterfactual_action_store as ca_store


class TestManifestSerialization:
    def test_round_trip(self):
        manifest = ca_store.CounterfactualActionStoreManifest(
            version="1.0",
            source_rlds_data_dir="gs://bucket/robocoin_bimanual",
            source_dataset_name="robocoin_bimanual",
            source_dataset_version="1.0.0",
            source_num_episodes=100,
            num_samples=32,
            action_dim=14,
            action_horizon=25,
            policy_config_name="robocoin_bimanual_pi05_rlds",
            policy_checkpoint_dir="/checkpoints/bc_flow",
            created_at="2026-01-01T00:00:00",
        )

        data = manifest.to_dict()
        restored = ca_store.CounterfactualActionStoreManifest.from_dict(data)

        assert restored.source_dataset_name == "robocoin_bimanual"
        assert restored.num_samples == 32
        assert restored.action_dim == 14
        assert restored.action_horizon == 25
        assert restored.policy_config_name == "robocoin_bimanual_pi05_rlds"

    def test_save_load(self):
        manifest = ca_store.CounterfactualActionStoreManifest(
            source_dataset_name="test_dataset",
            source_dataset_version="1.0.0",
            num_samples=8,
            action_dim=7,
            action_horizon=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ca_store.save_manifest(manifest, tmpdir)
            loaded = ca_store.load_manifest(tmpdir)

            assert loaded.source_dataset_name == "test_dataset"
            assert loaded.num_samples == 8
            assert loaded.action_dim == 7

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError):
            ca_store.load_manifest(tmpdir)


class TestManifestValidation:
    def test_matching_dataset(self):
        manifest = ca_store.CounterfactualActionStoreManifest(
            source_dataset_name="robocoin_bimanual",
            source_dataset_version="1.0.0",
        )
        ca_store.validate_manifest_against_rlds(
            manifest, rlds_data_dir="", dataset_name="robocoin_bimanual", dataset_version="1.0.0"
        )

    def test_mismatched_name_raises(self):
        manifest = ca_store.CounterfactualActionStoreManifest(
            source_dataset_name="robocoin_bimanual",
            source_dataset_version="1.0.0",
        )
        with pytest.raises(RuntimeError, match="dataset"):
            ca_store.validate_manifest_against_rlds(
                manifest, rlds_data_dir="", dataset_name="wrong_name", dataset_version="1.0.0"
            )

    def test_mismatched_version_raises(self):
        manifest = ca_store.CounterfactualActionStoreManifest(
            source_dataset_name="robocoin_bimanual",
            source_dataset_version="1.0.0",
        )
        with pytest.raises(RuntimeError, match="version"):
            ca_store.validate_manifest_against_rlds(
                manifest, rlds_data_dir="", dataset_name="robocoin_bimanual", dataset_version="2.0.0"
            )


@pytest.mark.manual
class TestFeatureSpec:
    def test_feature_spec_creation(self):
        manifest = ca_store.CounterfactualActionStoreManifest(
            num_samples=32,
            action_dim=14,
            action_horizon=25,
        )
        features = ca_store.get_counterfactual_action_store_tfds_feature_spec(manifest)

        assert "episode_index" in features
        assert "num_steps" in features
        assert "counterfactual_actions" in features


@pytest.mark.manual
class TestShardWriteRead:
    def test_write_and_read_episode(self):
        """Write an episode to a shard and read it back via TFDS builder."""
        import tensorflow_datasets as tfds

        manifest = ca_store.CounterfactualActionStoreManifest(
            source_dataset_name="test_dataset",
            source_dataset_version="1.0.0",
            num_samples=4,
            action_dim=7,
            action_horizon=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ca_store.CounterfactualActionStoreTFDSShardWriter(
                output_dir=tmpdir,
                shard_idx=0,
                manifest=manifest,
                split="train",
            )

            num_steps = 15
            actions = np.random.randn(num_steps, 4, 10, 7).astype(np.float32)
            writer.write_episode(
                episode_index=42,
                num_steps=num_steps,
                actions=actions,
            )
            writer.finalize()

            assert writer.episode_count == 1

            # Create dataset_info.json so we can load via builder
            features = ca_store.get_counterfactual_action_store_tfds_feature_spec(manifest)
            dataset_dir = writer.dataset_dir

            filename_template = tfds.core.ShardedFileTemplate(
                dataset_name=ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME,
                split="train",
                filetype_suffix="tfrecord",
                data_dir=str(dataset_dir),
                template="{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}",
            )

            split_info = tfds.core.SplitInfo(
                name="train",
                shard_lengths=[1],
                num_bytes=writer.total_bytes,
                filename_template=filename_template,
            )

            identity = tfds.core.DatasetIdentity(
                name=ca_store.COUNTERFACTUAL_ACTION_STORE_DATASET_NAME,
                version=tfds.core.Version(ca_store.VERSION),
                data_dir=str(dataset_dir),
                module_name=__name__,
            )
            info = tfds.core.DatasetInfo(
                builder=identity,
                features=features,
            )
            info.set_splits(tfds.core.SplitDict([split_info]))
            info.write_to_directory(dataset_dir)

            # Load back
            builder = tfds.builder_from_directory(str(dataset_dir))
            ds = builder.as_dataset(split="train")

            for example in ds:
                assert int(example["episode_index"]) == 42
                assert int(example["num_steps"]) == num_steps
                loaded_actions = example["counterfactual_actions"].numpy()
                assert loaded_actions.shape == (num_steps, 4, 10, 7)
                np.testing.assert_allclose(loaded_actions, actions, atol=1e-6)


class TestStridedActions:
    def test_stride_defaults_to_1(self):
        """Legacy manifests without stride field should default to stride=1."""
        data = {"source_dataset_name": "test", "source_dataset_version": "1.0.0"}
        manifest = ca_store.CounterfactualActionStoreManifest.from_dict(data)
        assert manifest.stride == 1

    def test_stride_serialization(self):
        """Test that stride is properly serialized and deserialized."""
        manifest = ca_store.CounterfactualActionStoreManifest(
            source_dataset_name="test",
            source_dataset_version="1.0.0",
            stride=5,
        )
        data = manifest.to_dict()
        assert data["stride"] == 5

        restored = ca_store.CounterfactualActionStoreManifest.from_dict(data)
        assert restored.stride == 5

    def test_expand_strided_actions(self):
        """Test expansion with stride=5."""
        import tensorflow as tf

        stride = 5
        num_steps = 20
        stored_action_horizon = 50
        target_action_horizon = 30
        num_samples = 3
        action_dim = 7

        # num_strided_positions = ceil(20 / 5) = 4
        num_strided_positions = 4

        # Shape: [num_strided_positions, num_samples, stored_action_horizon, action_dim]
        strided = tf.reshape(
            tf.range(
                num_strided_positions * num_samples * stored_action_horizon * action_dim,
                dtype=tf.float32,
            ),
            [num_strided_positions, num_samples, stored_action_horizon, action_dim],
        )

        expanded = ca_store.expand_strided_counterfactual_actions(strided, stride, num_steps, target_action_horizon)

        # Output: [num_steps, num_samples, target_action_horizon, action_dim]
        assert expanded.shape == (20, 3, 30, 7)

        # t=0: strided_idx=0, offset=0 -> slice [0:30] from stored_action_horizon dim
        np.testing.assert_array_equal(expanded[0], strided[0, :, 0:30, :])

        # t=3: strided_idx=0, offset=3 -> slice [3:33] from stored_action_horizon dim
        np.testing.assert_array_equal(expanded[3], strided[0, :, 3:33, :])

        # t=5: strided_idx=1, offset=0 -> slice [0:30] from stored_action_horizon dim
        np.testing.assert_array_equal(expanded[5], strided[1, :, 0:30, :])

        # t=7: strided_idx=1, offset=2 -> slice [2:32] from stored_action_horizon dim
        np.testing.assert_array_equal(expanded[7], strided[1, :, 2:32, :])

        # t=10: strided_idx=2, offset=0 -> slice [0:30] from stored_action_horizon dim
        np.testing.assert_array_equal(expanded[10], strided[2, :, 0:30, :])

        # t=19: strided_idx=3, offset=4 -> slice [4:34] from stored_action_horizon dim
        np.testing.assert_array_equal(expanded[19], strided[3, :, 4:34, :])

    def test_expand_strided_actions_stride_1(self):
        """Test that stride=1 is a no-op (just slices action horizon)."""
        import tensorflow as tf

        stride = 1
        num_steps = 10
        stored_action_horizon = 50
        target_action_horizon = 30
        num_samples = 3
        action_dim = 7

        strided = tf.reshape(
            tf.range(
                num_steps * num_samples * stored_action_horizon * action_dim,
                dtype=tf.float32,
            ),
            [num_steps, num_samples, stored_action_horizon, action_dim],
        )

        expanded = ca_store.expand_strided_counterfactual_actions(strided, stride, num_steps, target_action_horizon)

        assert expanded.shape == (10, 3, 30, 7)

        # For stride=1, each output is just the first target_action_horizon actions from the input
        for t in range(num_steps):
            np.testing.assert_array_equal(expanded[t], strided[t, :, 0:30, :])
