"""Tests for the ResNet-50 value network (tiny image size so ResNet-50 runs on CPU)."""

import dataclasses

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.model import Observation
from openpi.training import weight_loaders
from openpi.value_functions.heads import RegressionHeadConfig
from openpi.value_functions.networks import resnet as _resnet
from openpi.value_functions.value_function import CQLValueFunctionConfig
from openpi.value_functions.value_function import MCValueFunctionConfig

IMAGE_KEYS = ("cam_a", "cam_b")
IMAGE_SIZE = (64, 64)
BATCH = 2
ACTION_HORIZON = 5


def make_config(**overrides) -> _resnet.ResNetNetworkConfig:
    kwargs = {
        "state_dim": 4,
        "num_cameras": 2,
        "image_keys": IMAGE_KEYS,
        "image_size": IMAGE_SIZE,
        "action_dim": 3,
        "hidden_size": 32,
        "spatial_num_features": 2,
        "num_subtask_categories": 3,
        "subtask_vocab": ("a", "b", "c"),
    }
    kwargs.update(overrides)
    return _resnet.ResNetNetworkConfig(**kwargs)


def make_observation(*, subtask_id = None, action_mask = None, image_masks = None, seed = 86) -> Observation:
    keys = jax.random.split(jax.random.key(seed), len(IMAGE_KEYS))
    images = {
        key: jax.random.uniform(k, (BATCH, *IMAGE_SIZE, 3), minval = -1.0, maxval = 1.0)
        for key, k in zip(IMAGE_KEYS, keys, strict = True)
    }
    if image_masks is None:
        image_masks = {key: jnp.ones((BATCH,), dtype = bool) for key in IMAGE_KEYS}
    return Observation(
        images = images,
        image_masks = image_masks,
        state = jnp.ones((BATCH, 4)),
        action_mask = action_mask,
        subtask_id = subtask_id,
    )


@pytest.fixture(scope = "module")
def network():
    return make_config().create(jax.random.key(86), action_horizon = ACTION_HORIZON)


def test_feature_map_size():
    assert _resnet.feature_map_size((224, 224)) == (7, 7)
    assert _resnet.feature_map_size((64, 64)) == (2, 2)
    assert _resnet.feature_map_size((96, 128)) == (3, 4)


def test_config_validation():
    with pytest.raises(ValueError, match = "image_keys"):
        make_config(image_keys = ("only_one",))
    with pytest.raises(ValueError, match = "requires num_subtask_categories"):
        make_config(num_subtask_categories = None)
    with pytest.raises(ValueError, match = "entries"):
        make_config(num_subtask_categories = 2)
    with pytest.raises(ValueError, match = "duplicate"):
        make_config(subtask_vocab = ("a", "a.", "b"))
    assert make_config().get_tokenizer() is None
    assert make_config().predict_subtask_ar is False


def test_train_forward_returns_features_and_aux(network):
    obs = make_observation(subtask_id = jnp.array([0, 2]), action_mask = jnp.ones((BATCH, ACTION_HORIZON), dtype = bool))
    action = jnp.ones((BATCH, ACTION_HORIZON, 3))
    features, aux = network.compute_features(obs, action, rng = jax.random.key(0))
    assert features.shape == (BATCH, 32)
    assert network.feature_dim == 32
    assert aux["next_token_embeddings"].shape == (BATCH, 1, 32)
    assert aux["next_token_targets"].shape == (BATCH, 1)
    assert aux["next_token_mask"].shape == (BATCH, 1)
    assert network.decode(aux["next_token_embeddings"]).shape == (BATCH, 1, 3)


def test_train_without_subtask_id_raises(network):
    obs = make_observation()
    with pytest.raises(ValueError, match = "subtask_id"):
        network.compute_features(obs, jnp.ones((BATCH, ACTION_HORIZON, 3)), rng = jax.random.key(0))


def test_inference_uses_predicted_id_and_matches_prefix_cache(network):
    obs = make_observation()
    action = jnp.ones((BATCH, ACTION_HORIZON, 3))
    features = network.compute_features(obs, action)
    assert features.shape == (BATCH, 32)

    predicted = network.predict_subtask_id(obs)
    assert predicted.shape == (BATCH,)
    features_gt = network.compute_features(dataclasses.replace(obs, subtask_id = predicted), action)
    np.testing.assert_allclose(features, features_gt, atol = 1e-5)

    image_features, cached_id, _ = network.compute_prefix_cache(obs)
    assert image_features.shape == (BATCH, 2 * 32)
    np.testing.assert_array_equal(cached_id, predicted)
    cached_features = network.compute_features(obs, action, prefix_cache = (image_features, cached_id, None))
    np.testing.assert_allclose(features, cached_features, atol = 1e-5)
    assert network.prefix_cache_batch_axis == 0


def test_negative_subtask_id_falls_back_to_prediction(network):
    obs = make_observation()
    predicted = network.predict_subtask_id(obs)
    sentinel = jnp.full((BATCH,), -1, dtype = jnp.int32)
    _, resolved_id, _ = network.compute_prefix_cache(dataclasses.replace(obs, subtask_id = sentinel))
    np.testing.assert_array_equal(resolved_id, predicted)
    given = (predicted + 1) % network.config.num_subtask_categories
    _, resolved_id, _ = network.compute_prefix_cache(dataclasses.replace(obs, subtask_id = given))
    np.testing.assert_array_equal(resolved_id, given)


def test_masks_zero_contributions(network):
    obs = make_observation(subtask_id = jnp.array([1, 1]))
    action = jnp.ones((BATCH, ACTION_HORIZON, 3))
    zero_action = jnp.zeros_like(action)
    masked_obs = dataclasses.replace(obs, action_mask = jnp.zeros((BATCH, ACTION_HORIZON), dtype = bool))
    np.testing.assert_allclose(
        network.compute_features(masked_obs, action), network.compute_features(obs, zero_action), atol = 1e-5
    )
    image_features, _, _ = network.compute_prefix_cache(
        dataclasses.replace(obs, image_masks = {"cam_a": jnp.zeros((BATCH,), dtype = bool), "cam_b": jnp.ones((BATCH,), dtype = bool)})
    )
    np.testing.assert_array_equal(image_features[:, :32], 0.0)
    assert bool(jnp.any(image_features[:, 32:] != 0.0))


def test_v_network_without_subtasks():
    config = make_config(num_subtask_categories = None, subtask_vocab = None, no_state = False)
    net = config.create(jax.random.key(0))
    features = net.compute_features(make_observation(), rng = jax.random.key(1))
    assert not isinstance(features, tuple)
    assert features.shape == (BATCH, 32)
    assert not net.action_conditioned


def test_graphdef_matches_eval_shape(network):
    abstract = nnx.eval_shape(lambda: make_config().create(jax.random.key(86), action_horizon = ACTION_HORIZON))
    assert nnx.split(abstract)[0] == nnx.split(network)[0]


def test_param_paths_are_str_keyed_and_match_imagenet_loader(network, tmp_path):
    flat = traverse_util.flatten_dict(nnx.state(network).to_pure_dict(), sep = "/")
    assert all(isinstance(key, str) for key in flat)
    assert "encoder/stem_conv/kernel" in flat
    assert "encoder/layers/layer2/block0/downsample_conv/kernel" in flat
    assert "spatial_embedding/kernel" in flat
    assert not any(key.startswith("spatial_embeddings/") for key in flat)

    trunk_keys = {
        key.removeprefix("encoder/"): value
        for key, value in flat.items()
        if key.startswith("encoder/") and key.endswith("/kernel") and value is not None
    }
    assert len(trunk_keys) == 53
    npz_path = tmp_path / "resnet50_gn.npz"
    np.savez(npz_path, **{key: np.full(value.shape, 0.5, dtype = np.float32) for key, value in trunk_keys.items()})

    model_config = CQLValueFunctionConfig(
        q_network_config = make_config(), q_head_config = RegressionHeadConfig(), action_horizon = ACTION_HORIZON
    )
    params = nnx.state(model_config.create(jax.random.key(0))).to_pure_dict()
    loaded = weight_loaders.ResNet50ImageNetWeightLoader(str(npz_path)).load(params)
    flat_loaded = traverse_util.flatten_dict(loaded, sep = "/")
    for network_name in ("q_network", "target_q_network"):
        np.testing.assert_array_equal(flat_loaded[f"{network_name}/encoder/stem_conv/kernel"], 0.5)
    # Non-trunk leaves keep the reference values.
    flat_ref = traverse_util.flatten_dict(params, sep = "/")
    np.testing.assert_array_equal(flat_loaded["q_network/fc1/kernel"], flat_ref["q_network/fc1/kernel"])


def test_value_function_configs_create_and_compute_value():
    obs = make_observation(subtask_id = jnp.array([0, 1]))
    action = jnp.ones((BATCH, ACTION_HORIZON, 3))
    cql = CQLValueFunctionConfig(
        q_network_config = make_config(), q_head_config = RegressionHeadConfig(), action_horizon = ACTION_HORIZON
    )
    assert cql.weight_dtype == "float32"
    cql_model = cql.create(jax.random.key(0))
    assert cql_model.compute_value(obs, action).shape == (BATCH,)
    assert cql_model.compute_target_value(obs, action).shape == (BATCH,)

    mc = MCValueFunctionConfig(network_config = make_config(), head_config = RegressionHeadConfig())
    assert mc.create(jax.random.key(0)).compute_value(obs).shape == (BATCH,)


def test_sarsa_loss_adds_subtask_cross_entropy():
    from openpi.value_functions.base_value_functions import Transition
    from openpi.value_functions.value_function import SARSAValueFunctionConfig

    obs = make_observation(subtask_id = jnp.array([0, 2]), action_mask = jnp.ones((BATCH, ACTION_HORIZON), dtype = bool))
    transition = Transition(
        observation = obs,
        action = jnp.ones((BATCH, ACTION_HORIZON, 3)),
        reward = jnp.zeros(BATCH),
        next_observation = make_observation(subtask_id = jnp.array([0, 2]), seed = 7),
        next_action = jnp.ones((BATCH, ACTION_HORIZON, 3)),
        mc_return = jnp.ones(BATCH) * 0.5,
        termination = jnp.zeros(BATCH, dtype = bool),
        truncation = jnp.zeros(BATCH, dtype = bool),
        td_discount = None,
    )
    model = SARSAValueFunctionConfig(
        network_config = make_config(), head_config = RegressionHeadConfig(), action_horizon = ACTION_HORIZON,
        next_token_loss_weight = 0.0,
    ).create(jax.random.key(0))
    baseline_loss, _ = model.compute_loss(transition, rng = jax.random.key(3))
    model.next_token_loss_weight = 0.5
    loss, info = model.compute_loss(transition, rng = jax.random.key(3))
    assert "next_token_loss" in info
    assert info["next_token_loss"].shape == (BATCH,)
    np.testing.assert_allclose(loss - baseline_loss, 0.5 * info["next_token_loss"], atol = 1e-5)
